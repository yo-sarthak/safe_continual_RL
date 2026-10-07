#!/bin/bash

#SBATCH --job-name=eval_ckpts

#SBATCH --output=eval_ckpts_%j.log

#SBATCH --partition=mcs.gpu.q,tue.gpu2.q,tue.gpu3.q

#SBATCH --time=02:00:00

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=2

#SBATCH --mem-per-cpu=4G

#SBATCH --gres=gpu:1

#SBATCH --export=ALL

# Eval-only: run tb_condition_eval.py over an explicit list of already-saved
# checkpoints, at a common threshold grid. No training, no checkpoint writes.
#
# Required: CKPTLIST  path to a text file, one checkpoint dir per line
# Optional: ENV (safe_push_point), LVL (14), NORM (by_const), THRESH (25 15 10 5),
#           PADOBS
#
# NOTE: sbatch --export splits on commas, so never pass a comma-separated value
# through it. THRESH is space-separated here and converted to the comma form
# tb_condition_eval.py expects inside the job.

source ~/miniconda3/bin/activate crax

cd ~/crax2 || exit 1

export PYTHONPATH="$HOME/crax2:$PYTHONPATH"

: "${CKPTLIST:?set CKPTLIST}"
[ -f "$CKPTLIST" ] || { echo "!!! FATAL: no such CKPTLIST: $CKPTLIST"; exit 1; }

ENV=${ENV:-safe_push_point}; LVL=${LVL:-14}; NORM=${NORM:-by_const}
THRESH=${THRESH:-25 15 10 5}

export BUDGET_NORM=$NORM
export BUDGET_NORM_EVAL=$NORM
export BUDGET_SINK=0
export BUDGET_RESHAPE=0
export TB_EXEMPT_LAST2=1
export EVAL_THRESHOLDS=$(echo $THRESH | tr ' ' ',')
[ -n "${PADOBS:-}" ] && export PAD_OBS_TO=$PADOBS || unset PAD_OBS_TO

echo "### EVALCKPTS env=$ENV lvl=$LVL norm=$NORM thresholds=$EVAL_THRESHOLDS"

for c in $(grep -v '^[[:space:]]*$' "$CKPTLIST"); do
  echo ""
  echo "===== $c ====="
  if [ ! -f "${c}/ppo_network_config.json" ]; then echo "!!! MISSING: $c"; continue; fi
  python tb_condition_eval.py "$c" $ENV $LVL | tee /tmp/evc_$$.txt \
    | grep -E "threshold d|^ *[0-9]+\.[0-9]"
  awk -v c="$c" 'NF==5 && $1+0==$1 {print "@@EVC", c, "d" $1, $2, $3, $4}' /tmp/evc_$$.txt
done

echo ""
echo "eval_ckpts complete: $(date)"
