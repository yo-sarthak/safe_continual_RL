"""P3O + threshold-budget wrapper + optional multi-task Fisher EWC."""
import functools
import json
import os
import pickle

import jax
import jax.numpy as jnp

from brax.training.agents.p3o.train import train as p3o_train_fn
from brax.training.agents.p3o import losses as p3o_losses
from brax.envs.wrappers.threshold_budget import ThresholdBudgetWrapper
from brax.envs.wrappers.pad_obs import PadObsWrapper


def _apply_budget(env, threshold):
    _pad = os.environ.get('PAD_OBS_TO', '')
    if _pad:
        env = PadObsWrapper(env, int(_pad))
        print(f'[p3o_ewc] PadObsWrapper -> base obs padded to {_pad}')
    return ThresholdBudgetWrapper(env, threshold=threshold)


def _load_ewc_tasks():
    """Load multi-task EWC list from EWC_PREV_TASKS, else legacy single task."""
    prev_tasks_json = os.environ.get("EWC_PREV_TASKS", "").strip()

    if prev_tasks_json:
        raw = json.loads(prev_tasks_json)
        tasks = []
        for t in raw:
            fisher_path = t["fisher"]
            coef = float(t.get("coef", 0.0))
            fisher_key_a = t.get("fisher_key_a", t.get("fisher_key", "F_vanilla"))
            fisher_key_b = t.get("fisher_key_b", t.get("fisher_key", "F_vanilla"))
            alpha = float(t.get("alpha", 1.0))
            beta = float(t.get("beta", 0.0))

            with open(fisher_path, "rb") as f:
                d = pickle.load(f)

            to_jnp = lambda x: jax.tree_util.tree_map(jnp.asarray, x)
            policy_star = to_jnp(d["policy_star"])

            def _pick(key):
                if key == "ONES":
                    return jax.tree_util.tree_map(jnp.ones_like, policy_star)
                return to_jnp(d[key])

            tasks.append({
                "policy_star": policy_star,
                "F_reward": _pick(fisher_key_a),
                "F_cost": _pick(fisher_key_b),
                "alpha": alpha,
                "beta": beta,
                "coef": coef,
            })

        return tasks

    # Legacy single-task EWC from phase2.sh
    coef = float(os.environ.get("EWC_COEF", "0"))
    fisher_path = os.environ.get("EWC_FISHER_PATH", "")

    if coef > 0 and fisher_path:
        with open(fisher_path, "rb") as f:
            d = pickle.load(f)

        to_jnp = lambda t: jax.tree_util.tree_map(jnp.asarray, t)

        def _pk(envkey, default):
            name = os.environ.get(envkey, default)
            if name == "ONES":
                return jax.tree_util.tree_map(jnp.ones_like, to_jnp(d["policy_star"]))
            return to_jnp(d[name])

        return [{
            "policy_star": to_jnp(d["policy_star"]),
            "F_reward": _pk("EWC_FISHER_A", "F_reward"),
            "F_cost": _pk("EWC_FISHER_B", "F_cost"),
            "alpha": float(os.environ.get("EWC_ALPHA", "0.0")),
            "beta": float(os.environ.get("EWC_BETA", "1.0")),
            "coef": coef,
        }]

    return None


@functools.wraps(p3o_train_fn)
def train(environment, *args, **kwargs):
    safety_bound = kwargs.get("safety_bound", 25.0)
    wrap_env = kwargs.get("wrap_env", True)

    tasks = _load_ewc_tasks()

    if tasks:
        p3o_losses.set_ewc(tasks)
        print(f"[p3o_ewc] EWC ON with {len(tasks)} task(s)")
        for i, t in enumerate(tasks):
            print(f"  task {i}: coef={t['coef']} alpha={t['alpha']} beta={t['beta']}")
    else:
        p3o_losses.set_ewc(None)
        print("[p3o_ewc] EWC OFF (naive continue)")

    restore = os.environ.get("RESTORE_CKPT", "")
    if restore:
        kwargs.setdefault("restore_checkpoint_path", restore)
        print(f"[p3o_ewc] restoring from {restore}")

    if wrap_env and environment is not None:
        environment = _apply_budget(environment, safety_bound)

    eval_env = kwargs.get("eval_env", None)
    if eval_env is not None and wrap_env:
        kwargs["eval_env"] = _apply_budget(eval_env, safety_bound)

    kwargs["environment"] = environment
    return p3o_train_fn(**kwargs)