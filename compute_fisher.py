"""Compute policy-network diagonal Fisher importances for dual-Fisher EWC.

Produces TWO importance maps over the POLICY network's parameters, from a phase-1
rollout of the trained anchor:

  F_reward : weighted by the (centered, Monte-Carlo) discounted REWARD-to-go
  F_cost   : weighted by the (centered, Monte-Carlo) discounted COST-to-go

Both are diagonal empirical Fishers of the policy:  F_i = mean_j ( adv_j * dlogpi_j/dtheta_i )^2 ,
computed over policy parameters only (value/cost critics are deliberately NOT
protected -- they re-fit each phase). Cost-to-go is Monte-Carlo because CRAX's
checkpoint saves (normalizer, policy, value) but NOT the cost critic.

It saves {policy_star, F_reward, F_cost, meta} to a pickle, and prints an
OVERLAP DIAGNOSTIC (cosine similarity + top-k parameter overlap between F_reward
and F_cost). That diagnostic is the cheap pre-check for whether dual Fisher has
any room to move on this axis: if F_reward and F_cost are nearly identical, the
alpha/beta knob is inert here (expected on the pure-threshold axis); if they
diverge, dual Fisher is doing real work (expected on difficulty/task axes).

USAGE:
    python compute_fisher.py <phase1_ckpt_dir> <phase1_d> <out.pkl>
    # BUDGET_NORM_EVAL env var must match how the anchor was trained
    #   ("by_threshold" for _d anchors, "by_const" for _25 anchors)

Run on a GPU node.
"""
import os
import sys
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from brax import envs
from brax.envs.wrappers import training as tw
from brax.envs.wrappers.threshold_budget import ThresholdBudgetWrapper
from brax.envs.wrappers.pad_obs import PadObsWrapper
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks

# ----------------------------- config -----------------------------
ENV_NAME = "safe_goal_point"
LEVEL = 1
EPISODE_LENGTH = 1000
NUM_ENVS = 128
ROLLOUT_STEPS = 400            # NUM_ENVS * ROLLOUT_STEPS samples for the Fisher (~51k)
DISCOUNT = 0.99                # must match training 'discounting'
CHUNK = 2048                   # samples per grad chunk (memory knob)
NORM_IMPORTANCE = True         # divide each Fisher by its mean (norm_importance=True)
SEED = 0
# ------------------------------------------------------------------

if len(sys.argv) < 4:
    sys.exit("usage: python compute_fisher.py <phase1_ckpt_dir> <phase1_d> <out.pkl>")
ckpt_dir = sys.argv[1]
phase1_d = float(sys.argv[2])
out_path = sys.argv[3]
# env/level must match the anchor. Previously hardcoded to Goal at line ~43,
# which silently broke every non-Goal Fisher (obs-dim mismatch).
ENV_NAME = sys.argv[4] if len(sys.argv) > 4 else ENV_NAME
LEVEL    = int(sys.argv[5]) if len(sys.argv) > 5 else LEVEL
# episode length must match training (Goal=1000, Push=2000, Reacher=200);
# hardcoding 1000 truncates Push rollouts and never reaches the low-budget states.
import brax.envs as _envs
EPISODE_LENGTH = int(sys.argv[6]) if len(sys.argv) > 6 else int(
    getattr(_envs.get_environment(ENV_NAME, level=LEVEL), 'episode_length', EPISODE_LENGTH))
print(f'[fisher] env={ENV_NAME} level={LEVEL} d={phase1_d} episode_length={EPISODE_LENGTH}')
budget_norm = os.environ.get("BUDGET_NORM_EVAL", "by_threshold")

# --- load config + params, build networks; neutralize normalizer on last 2 dims ---
cfg = ppo_checkpoint.load_config(ckpt_dir)
params = ppo_checkpoint.load(ckpt_dir)               # (normalizer, policy, value)
net = ppo_checkpoint._get_ppo_network(cfg, ppo_networks.make_ppo_networks)

norm = params[0]
norm = norm.replace(mean=norm.mean.at[-2:].set(0.0), std=norm.std.at[-2:].set(1.0))
policy_star = params[1]                               # the params we will protect
# --- TRUE cost critic, if this checkpoint was saved with one (4-tuple) ---
# P3O builds its cost critic with cost_value_hidden_layer_sizes=(256,)*5
# (see brax/training/agents/p3o/train.py). The saved config does not record
# that, so we rebuild the network with the same literal to apply the params.
cost_value_star = params[3] if len(params) > 3 else None
cost_value_apply = None
if cost_value_star is not None:
    import functools as _ft
    _nf = _ft.partial(ppo_networks.make_ppo_networks,
                      cost_value_hidden_layer_sizes=(256,) * 5)
    _net_cv = ppo_checkpoint._get_ppo_network(cfg, _nf)
    cost_value_apply = _net_cv.cost_value_network.apply
    print('[fisher] TRUE cost critic loaded from checkpoint')
else:
    print('[fisher] no cost critic in checkpoint (3-tuple) -- V_C baseline unavailable')
dist = net.parametric_action_distribution
policy_apply = net.policy_network.apply

# stochastic inference (sample actions) so Fisher reflects the policy distribution
make_inf = ppo_networks.make_inference_fn(net)
infer = jax.jit(make_inf((norm, policy_star), deterministic=False))

# --- build phase-1 eval env at phase1_d ---
base = envs.get_environment(ENV_NAME, level=LEVEL)
# mirror the TRAINING wrapper order: pad the base obs, then add budget channels
_pad = os.environ.get('PAD_OBS_TO', '')
if _pad:
    base = PadObsWrapper(base, int(_pad))
    print(f'[fisher] PadObsWrapper -> base padded to {_pad}')
env = ThresholdBudgetWrapper(base, threshold=phase1_d, budget_norm=budget_norm)
env = tw.wrap(env, episode_length=EPISODE_LENGTH, action_repeat=1)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# --- roll out, collecting obs, sampled raw_action, reward, cost, done ---
key = jax.random.PRNGKey(SEED)
key, rk = jax.random.split(key)
state = jit_reset(jax.random.split(rk, NUM_ENVS))

obs_buf, act_buf, rew_buf, cost_buf, done_buf = [], [], [], [], []
for _ in range(ROLLOUT_STEPS):
    key, ak = jax.random.split(key)
    action, extra = infer(state.obs, ak)
    raw_action = extra["raw_action"]                 # matches log_prob param in the loss
    obs_buf.append(state.obs)
    act_buf.append(raw_action)
    nstate = jit_step(state, action)
    rew_buf.append(nstate.reward)
    cost_buf.append(nstate.metrics["cost"])
    done_buf.append(nstate.done)
    state = nstate

obs = jnp.stack(obs_buf)          # [T, B, obs]
act = jnp.stack(act_buf)          # [T, B, act]
rew = jnp.stack(rew_buf)          # [T, B]
cost = jnp.stack(cost_buf)        # [T, B]
done = jnp.stack(done_buf)        # [T, B]


def discounted_to_go(x, done, gamma):
    """Reverse discounted cumulative sum along time, resetting at done. x:[T,B]."""
    def body(carry, inp):
        xt, dt = inp
        carry = xt + gamma * carry * (1.0 - dt)
        return carry, carry
    _, out = jax.lax.scan(body, jnp.zeros(x.shape[1]), (x[::-1], done[::-1]))
    return out[::-1]

G_r = discounted_to_go(rew, done, DISCOUNT)          # [T,B] reward-to-go
G_c = discounted_to_go(cost, done, DISCOUNT)         # [T,B] cost-to-go

# flatten and center each advantage (centering -> zero-mean importance weights)
obs_f = obs.reshape(-1, obs.shape[-1])
act_f = act.reshape(-1, act.shape[-1])
adv_r = (G_r.reshape(-1) - G_r.mean())
adv_c = (G_c.reshape(-1) - G_c.mean())
# VANILLA (Kirkpatrick/Coursey PPO+EWC): unit weights -> the TRUE policy Fisher,
#   F = E[(dlogpi/dtheta)^2] ~ Hessian of KL(pi_theta* || pi_theta). Objective-agnostic.
adv_v = jnp.ones_like(adv_r)
# COURSEY CF-EWC: per-sample weight 1/(1+c_n) on INSTANTANEOUS cost. Down-weights
#   samples that were unsafe -> releases those weights. Opposite sign to our F_cost.
#   NOTE: sqrt because fisher_over squares the weight.
c_inst = cost.reshape(-1)
adv_ccf = jnp.sqrt(1.0 / (1.0 + c_inst))
# --- SURROGATE STATE-DEPENDENT BASELINE (screen only) ---
# Textbook advantage is A_C = G_C - V_C(s), but P3O's cost critic is not saved in
# the checkpoint. Here we FIT a cheap linear predictor of G_c from the rollout's own
# observations and use it as V_C. It is not the trained critic -- it only answers
# 'does a state-dependent baseline change the Fisher at all?'. If cos(F_cost,
# F_cost_vb) ~ 1 the refinement is inert and the trained critic is not worth
# retraining every anchor for.
import numpy as _np
_X = _np.asarray(obs_f); _y = _np.asarray(G_c.reshape(-1))
_X1 = _np.concatenate([_X, _np.ones((_X.shape[0], 1), dtype=_X.dtype)], axis=1)
_w, *_ = _np.linalg.lstsq(_X1, _y, rcond=None)
_vpred = _X1 @ _w
_r2 = 1.0 - _np.var(_y - _vpred) / (_np.var(_y) + 1e-12)
print(f'[baseline] linear V_C fit R^2 = {_r2:.3f}  (higher = baseline explains more of G_c)')
adv_cvb = jnp.asarray(_y - _vpred)
print(f"[coursey] frac samples with c_n==0: {float((c_inst==0).mean()):.3f}  "
      f"weight min={float(adv_ccf.min()**2):.3f} mean={float((adv_ccf**2).mean()):.3f}")
N = obs_f.shape[0]
print(f"collected {N} samples  |  adv_r std={float(adv_r.std()):.3f}  adv_c std={float(adv_c.std()):.3f}")

# --- TEXTBOOK cost advantage: A_C = G_C - V_C(s) using the TRAINED critic ---
adv_creal = None
if cost_value_apply is not None:
    _vc = []
    for _s in range(0, N, CHUNK):
        _vc.append(jnp.ravel(cost_value_apply(norm, cost_value_star, obs_f[_s:_s+CHUNK])))
    _vc = jnp.concatenate(_vc)
    _yj = jnp.asarray(_y)
    _r2r = 1.0 - float(jnp.var(_yj - _vc) / (jnp.var(_yj) + 1e-12))
    print(f'[baseline] TRAINED V_C fit R^2 = {_r2r:.3f}')
    adv_creal = _yj - _vc


def per_sample_sqgrad(policy_params, o_chunk, a_chunk, adv_chunk):
    """Sum over the chunk of (adv * dlogpi/dtheta)^2, as a param-shaped tree."""
    def single(o, a, adv):
        def f(pp):
            logits = policy_apply(norm, pp, o[None])
            return adv * dist.log_prob(logits, a[None])[0]
        return jax.grad(f)(policy_params)
    gs = jax.vmap(single)(o_chunk, a_chunk, adv_chunk)     # tree, leading dim = chunk
    return jax.tree_util.tree_map(lambda g: jnp.sum(g * g, axis=0), gs)

per_sample_sqgrad = jax.jit(per_sample_sqgrad)


def fisher_over(adv_flat):
    acc = jax.tree_util.tree_map(jnp.zeros_like, policy_star)
    n = 0
    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)
        chunk = per_sample_sqgrad(policy_star, obs_f[s:e], act_f[s:e], adv_flat[s:e])
        acc = jax.tree_util.tree_map(lambda a, c: a + c, acc, chunk)
        n += (e - s)
    fisher = jax.tree_util.tree_map(lambda a: a / n, acc)
    if NORM_IMPORTANCE:
        flat = jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(fisher)])
        m = jnp.mean(flat) + 1e-12
        fisher = jax.tree_util.tree_map(lambda a: a / m, fisher)
    return fisher

print("computing F_reward ...")
F_reward = fisher_over(adv_r)
print("computing F_cost ...")
F_cost = fisher_over(adv_c)
print("computing F_vanilla ...")
F_vanilla = fisher_over(adv_v)
print("computing F_coursey_cf ...")
F_coursey = fisher_over(adv_ccf)
print('computing F_cost_vb (state-dependent baseline) ...')
F_cost_vb = fisher_over(adv_cvb)
F_cost_realvb = None
if adv_creal is not None:
    print('computing F_cost_realvb (TRAINED critic baseline) ...')
    F_cost_realvb = fisher_over(adv_creal)

# --- overlap diagnostic: do the two importance maps actually differ? ---
fr = np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(F_reward)])
fc = np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(F_cost)])
cos = float(fr @ fc / (np.linalg.norm(fr) * np.linalg.norm(fc) + 1e-12))
k = max(1, int(0.1 * fr.size))
top_r = set(np.argsort(fr)[-k:]); top_c = set(np.argsort(fc)[-k:])
overlap = len(top_r & top_c) / k
print("\n================ DUAL-FISHER SEPARATION ================")
print(f"cosine(F_reward, F_cost) = {cos:.3f}   (1.0 = identical; low = they protect different weights)")
print(f"top-10% parameter overlap = {overlap:.3f}   (1.0 = same params matter; low = distinct)")
print("high cosine/overlap -> alpha/beta knob is ~inert on this axis (expected for pure-threshold)")
print("low  cosine/overlap -> dual Fisher separates -> knob is meaningful (expected for difficulty/task)")
print("=======================================================\n")
# pairwise cosine / top-decile overlap across ALL variants -> decide which are
# distinct enough to be worth 5 seeds of phase-2 training.
_maps = {"vanilla": F_vanilla, "ours_cf": F_cost, "reward": F_reward, "coursey_cf": F_coursey, "ours_cf_vb": F_cost_vb}
if F_cost_realvb is not None: _maps["ours_cf_realvb"] = F_cost_realvb
_flat = {k: np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(v)])
         for k, v in _maps.items()}
# CONCENTRATION: share of total Fisher mass in the top 1% of parameters.
# Advantage-weighted maps are second moments of the policy gradient, so a few
# high-advantage samples can dominate. High concentration => fragile estimate.
print("======== FISHER MASS CONCENTRATION ========")
for _k, _v in _maps.items():
    _a = np.concatenate([np.ravel(np.asarray(x)) for x in jax.tree_util.tree_leaves(_v)])
    _srt = np.sort(_a)[::-1]; _tot = _srt.sum() + 1e-12
    print(f"@@FMASS {_k:12s} top1%={_srt[:max(1,int(0.01*_a.size))].sum()/_tot:.3f}  top10%={_srt[:max(1,int(0.10*_a.size))].sum()/_tot:.3f}")
print("")
print("======== FISHER VARIANT SIMILARITY (cosine | top-10% overlap) ========")
_ks = list(_flat)
for _i in range(len(_ks)):
    for _j in range(_i+1, len(_ks)):
        a, b = _flat[_ks[_i]], _flat[_ks[_j]]
        _cos = float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
        _k2 = max(1, int(0.1*a.size))
        _ov = len(set(np.argsort(a)[-_k2:]) & set(np.argsort(b)[-_k2:])) / _k2
        print(f"@@FCMP {_ks[_i]:10s} vs {_ks[_j]:10s}  cos={_cos:.3f}  overlap={_ov:.3f}")
print("=====================================================================\n")

with open(out_path, "wb") as f:
    pickle.dump({
        "policy_star": jax.tree_util.tree_map(np.asarray, policy_star),
        "F_reward": jax.tree_util.tree_map(np.asarray, F_reward),
        "F_cost": jax.tree_util.tree_map(np.asarray, F_cost),
        "F_vanilla": jax.tree_util.tree_map(np.asarray, F_vanilla),
        "F_coursey": jax.tree_util.tree_map(np.asarray, F_coursey),
        "F_cost_vb": jax.tree_util.tree_map(np.asarray, F_cost_vb),
        "F_cost_realvb": (jax.tree_util.tree_map(np.asarray, F_cost_realvb)
                          if F_cost_realvb is not None else None),
        "meta": {"phase1_d": phase1_d, "budget_norm": budget_norm,
                  "num_samples": int(N), "cosine": cos, "top10_overlap": overlap},
    }, f)
print(f"saved -> {out_path}")
