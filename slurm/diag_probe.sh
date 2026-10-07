#!/bin/bash
#SBATCH --job-name=diag_probe
#SBATCH --output=diag_%A_%a.log
#SBATCH --partition=mcs.gpu.q,tue.gpu2.q,tue.gpu3.q
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gpus=1
#SBATCH --export=ALL
#
# ============================================================================
# DIAGNOSTIC PROBE — one (task, substrate, threshold) cell per job, 500M steps.
#
# Answers two INDEPENDENT within-task questions:
#   (A) tradeoff:  within this task, does reward move as d changes?  (compare
#                  the d=10/15/25 jobs of the SAME task+substrate to each other)
#   (B) substrate: does sink vs no-sink stay stable?  (compare cost_std/cost
#                  between the sink and no-sink jobs at the SAME task+d)
#
# Launch all 8 cells (see launch_diag.sh):
#   ENV=safe_push_point  SUB=sink    D in {10,15,25}
#   ENV=safe_push_point  SUB=nosink  D in {10,15,25}
#   ENV=safe_goal_point  SUB=nosink  D in {10,25}   DIFF=2
#
# Required exports: ENV, D, SUB.   Optional: DIFF (default 1), SEED (default 0),
#                   STEPS (default 500e6), TAG (default auto).
# ============================================================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB_MODE=online
source ~/miniconda3/bin/activate crax
cd ~/crax2 || { echo "crax2 not found"; exit 1; }
export PYTHONPATH="$HOME/crax2:$PYTHONPATH"

: "${ENV:?set ENV, e.g. safe_push_point or safe_goal_point}"
: "${D:?set D (safety bound), e.g. 10 / 15 / 25}"
: "${SUB:?set SUB=sink or SUB=nosink}"
ALG=${ALG:-p3o_budget}     # p3o_budget = wrapped(+2 obs); p3o = native unwrapped control
NORM=${NORM:-by_threshold}
HPTAG=${HPTAG:-}     # label for a hyperparameter variant, e.g. gamma995
LR=${LR:-}; ENT=${ENT:-}; GAMMA=${GAMMA:-}; UNROLL=${UNROLL:-}; NUPD=${NUPD:-}
DIFF=${DIFF:-1}
SEED=${SEED:-0}
STEPS=${STEPS:-500e6}
ENVS=2048; EVALS=5
TAG=${TAG:-diag_${ENV}_L${DIFF}_${ALG}_${SUB}_${NORM}${HPTAG:+_$HPTAG}_d${D}_s${SEED}}
WPROJ=crax-cl-testing
WGROUP=diag_${ENV}_L${DIFF}
MODELS=models/${TAG}
WRAPPER="brax/envs/wrappers/threshold_budget.py"
mkdir -p "$MODELS"

export BUDGET_NORM=$NORM
export BUDGET_NORM_EVAL=$NORM
# exempt the 2 wrapper channels from running normalization (wrapped algs only)
[ "$ALG" = "p3o_budget" ] && export TB_EXEMPT_LAST2=1
[ "$ALG" = "p3o_budget" ] || unset TB_EXEMPT_LAST2
if [ -n "${PADOBS:-}" ]; then export PAD_OBS_TO=$PADOBS; else unset PAD_OBS_TO; fi
echo "### TB_EXEMPT_LAST2=${TB_EXEMPT_LAST2:-<unset>}  (must be 1 for p3o_budget runs)"

# ---- substrate selection ----
unset BUDGET_SINK BUDGET_RESHAPE RESHAPE_SCALE RESHAPE_CAP
case "$SUB" in
  sink)   export BUDGET_SINK=1; export BUDGET_RESHAPE=0
          echo "### substrate: HARD SINK" ;;
  nosink) export BUDGET_SINK=0; export BUDGET_RESHAPE=0
          echo "### substrate: NO-SINK (reward untouched; kappa handles constraint)" ;;
  *) echo "!!! FATAL: SUB must be 'sink' or 'nosink' (got '$SUB')"; exit 1 ;;
esac

scontrol update JobId=$SLURM_JOB_ID Name="L${DIFF}_d${D}_s${SEED}_${HPTAG:-stock}" 2>/dev/null
echo "############################################################"
echo "# DIAG  env=${ENV}  L${DIFF}  alg=${ALG}  substrate=${SUB}  d=${D}  seed=${SEED}  steps=${STEPS}"
echo "############################################################"

# ---- train ----
# per-job --model_dir removes the need for a before/after guard
WANDB_RUN_GROUP="$WGROUP" python train_env.py \
  --alg $ALG --env_name $ENV --difficulty $DIFF \
  --model_dir models_${TAG} \
  ${EPLEN:+--episode_length $EPLEN} \
  ${LR:+--learning_rate $LR} ${ENT:+--entropy_cost $ENT} ${GAMMA:+--discounting $GAMMA} ${UNROLL:+--unroll_length $UNROLL} ${NUPD:+--num_updates_per_batch $NUPD} \
  --safety_bound "$D" --num_envs $ENVS --num_timesteps $STEPS \
  --num_evals $EVALS --num_eval_envs 128 --deterministic_eval True \
  --skip_rollout --skip_video --seeds $SEED \
  --wandb_project $WPROJ --wandb_group $WGROUP --wandb_tags "${TAG: -60}"

newest=$(ls -td models_${TAG}/*/ 2>/dev/null | head -1)
[ -n "$newest" ] || { echo "!!! FATAL: no ckpt under models_${TAG}"; exit 1; }
ckpt=$(ls -d "${newest}"[0-9]*/ 2>/dev/null | sort | tail -1)
[ -n "$ckpt" ] && [ -f "${ckpt}ppo_network_config.json" ] || { echo "!!! FATAL: bad ckpt under $newest"; exit 1; }
rm -rf "$MODELS/anchor"; cp -r "$ckpt" "$MODELS/anchor"
echo "### saved -> $MODELS/anchor"

# ---- eval: TRUE objective (substrate OFF), threshold sweep ----
echo "########## EVAL (substrate OFF = true reward/cost) ##########"
BUDGET_SINK=0 BUDGET_RESHAPE=0 python tb_condition_eval.py "$MODELS/anchor" $ENV $DIFF \
  | tee /tmp/eval_${TAG}.txt

# ---- one-line machine-readable summary for fast scanning across all 8 jobs ----
# pull the row matching THIS job's training threshold d
row=$(awk -v d="$D" '$1+0==d+0 {print $2, $3, $4}' /tmp/eval_${TAG}.txt | head -1)
rew=$(echo $row | awk '{print $1}')
cost=$(echo $row | awk '{print $2}')
cstd=$(echo $row | awk '{print $3}')
echo ""
echo "@@SUMMARY ${ENV} L${DIFF} ${SUB} d=${D} seed=${SEED} | reward=${rew} cost=${cost} cost_std=${cstd}"
python3 - <<PY
d=float("$D"); 
try:
    c=float("$cost"); s=float("$cstd"); r=float("$rew")
except: 
    print("@@FLAGS  could not parse eval row -- check log"); raise SystemExit
flags=[]
if c > 1.2*d: flags.append(f"OVER-BOUND (cost {c:.1f} > 1.2*d={1.2*d:.1f})")
if s > c:     flags.append(f"UNSTABLE (cost_std {s:.1f} > cost {c:.1f})")
if not flags: flags.append("OK (on-bound, variance sane)")
print("@@FLAGS  " + " ; ".join(flags))
PY
echo "diag $TAG complete: $(date)"
