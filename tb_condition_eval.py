"""Threshold-conditioning eval for a trained budget agent.

USAGE:
    python tb_condition_eval.py <checkpoint_dir> [env_name] [level]
Defaults: env_name=safe_goal_point, level=1  (existing Goal calls unchanged).
For Push:    python tb_condition_eval.py <ckpt> safe_push_point 1
For Goal L2: python tb_condition_eval.py <ckpt> safe_goal_point 2
"""
import sys
import os
import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.wrappers import training as tw
from brax.envs.wrappers.threshold_budget import ThresholdBudgetWrapper
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks

BUDGET_NORM_EVAL = os.environ.get("BUDGET_NORM_EVAL", "by_threshold")
EPISODE_LENGTH = 1000
NUM_EVAL_ENVS = 128
THRESHOLDS = [25.0, 15.0, 10.0, 5.0]
SEED = 0

if len(sys.argv) < 2:
    sys.exit("usage: python tb_condition_eval.py <checkpoint_dir> [env_name] [level]")
ckpt_dir = sys.argv[1]
ENV_NAME = sys.argv[2] if len(sys.argv) > 2 else "safe_goal_point"
LEVEL = int(sys.argv[3]) if len(sys.argv) > 3 else 1

cfg = ppo_checkpoint.load_config(ckpt_dir)
params = ppo_checkpoint.load(ckpt_dir)
ppo_net = ppo_checkpoint._get_ppo_network(cfg, ppo_networks.make_ppo_networks)
make_inf = ppo_networks.make_inference_fn(ppo_net)

norm = params[0]
mean = norm.mean.at[-2:].set(0.0)
std = norm.std.at[-2:].set(1.0)
params = (norm.replace(mean=mean, std=std),) + tuple(params[1:])

inference_fn = make_inf(params, deterministic=True)
jit_infer = jax.jit(inference_fn)


def eval_at_threshold(d, key):
    base = envs.get_environment(ENV_NAME, level=LEVEL)
    env = ThresholdBudgetWrapper(base, threshold=d, budget_norm=BUDGET_NORM_EVAL)
    env = tw.wrap(env, episode_length=EPISODE_LENGTH, action_repeat=1)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    key, rk = jax.random.split(key)
    state = jit_reset(jax.random.split(rk, NUM_EVAL_ENVS))
    active = jnp.ones(NUM_EVAL_ENVS)
    cost_acc = jnp.zeros(NUM_EVAL_ENVS)
    rew_acc = jnp.zeros(NUM_EVAL_ENVS)
    for _ in range(EPISODE_LENGTH):
        key, ak = jax.random.split(key)
        action, _ = jit_infer(state.obs, ak)
        state = jit_step(state, action)
        cost_acc = cost_acc + state.metrics["cost"] * active
        rew_acc = rew_acc + state.reward * active
        active = active * (1.0 - state.done)
    return (float(jnp.mean(rew_acc)), float(jnp.mean(cost_acc)),
            float(jnp.std(cost_acc)), key)


key = jax.random.PRNGKey(SEED)
print(f"checkpoint: {ckpt_dir}")
print(f"env={ENV_NAME} level={LEVEL} episode_length={EPISODE_LENGTH} num_envs={NUM_EVAL_ENVS}\n")
print(f"{'threshold d':>11} {'reward':>9} {'cost':>9} {'cost_std':>9} {'cost/d':>8}")
for d in THRESHOLDS:
    r, c, cs, key = eval_at_threshold(d, key)
    print(f"{d:>11.1f} {r:>9.2f} {c:>9.2f} {cs:>9.2f} {c / d:>8.2f}")
print("\nRead the 'cost' column top to bottom:")
print("  falling with d (ideally cost ~ d)  -> the policy IS conditioning")
print("  flat near ~25 regardless of d      -> the policy IGNORES the threshold")
