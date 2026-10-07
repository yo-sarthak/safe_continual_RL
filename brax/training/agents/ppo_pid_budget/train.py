"""PPO-PID + threshold-budget trainer shim.

Pre-wraps the environment with ThresholdBudgetWrapper (Saute Piece 1: depleting
budget obs + static threshold obs, no reward reshaping, no early termination) and
delegates to the validated PPO-PID trainer. The PID Lagrangian handles the
constraint; the wrapper only conditions the policy on remaining budget + threshold.

The threshold `d` used by the wrapper is `safety_bound` -- the same number PPO-PID
regulates toward -- so conditioning and constraint stay consistent across phases.

Two implementation details that make this work with CRAX's dispatch:
  1. We import the base `train` FUNCTION directly from the submodule. The package
     `ppo_pid` re-exports `train` as the function, so `from ...ppo_pid import train`
     would bind the function (not the module) and `.train` on it would fail.
  2. run_utils.filter_kwargs_for_fn keeps only cfg keys whose names appear in this
     function's signature. A bare **kwargs would therefore drop num_envs,
     num_timesteps, etc. We set __signature__ to the base trainer's signature so
     every training arg is forwarded, while still accepting them via **kwargs.
"""
import inspect
from typing import Any, Optional

from brax import envs
from brax.training.agents.ppo_pid.train import train as _base_train
from brax.envs.wrappers.threshold_budget import ThresholdBudgetWrapper


def _apply_budget(env: envs.Env, threshold: float) -> envs.Env:
    # Scheme knobs (normalization, depletion) are edited in threshold_budget.py.
    return ThresholdBudgetWrapper(env, threshold=threshold)


def train(environment: envs.Env, eval_env: Optional[envs.Env] = None,
          wrap_env: bool = True, **kwargs: Any):
    safety_bound = kwargs.get('safety_bound', 25.0)
    if wrap_env:
        environment = _apply_budget(environment, safety_bound)
        if eval_env is not None:
            eval_env = _apply_budget(eval_env, safety_bound)

    return _base_train(environment=environment, eval_env=eval_env,
                       wrap_env=wrap_env, **kwargs)


# Advertise the base trainer's parameters so run_utils.filter_kwargs_for_fn
# forwards num_envs / num_timesteps / num_evals / seed / etc. into **kwargs.
train.__signature__ = inspect.signature(_base_train)
