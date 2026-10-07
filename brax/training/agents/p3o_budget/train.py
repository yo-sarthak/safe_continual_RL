"""P3O + threshold-budget trainer shim.

Pre-wraps the environment with ThresholdBudgetWrapper (Saute Piece 1: depleting
budget obs + static threshold obs, no reward reshaping, no early termination) and
delegates to the validated P3O trainer. The P3O penalizer (kappa) handles the
constraint; the wrapper only conditions the policy on remaining budget + threshold.

The threshold `d` used by the wrapper is `safety_bound` -- the same number P3O
regulates toward -- so conditioning and constraint stay consistent across phases.

Two implementation details that make this work with CRAX's dispatch:
  1. We import the base `train` FUNCTION directly from the submodule. The package
     `p3o` re-exports `train` as the function, so `from ...p3o import train`
     would bind the function (not the module) and `.train` on it would fail.
  2. run_utils.filter_kwargs_for_fn keeps only cfg keys whose names appear in this
     function's signature. A bare **kwargs would therefore drop num_envs,
     num_timesteps, etc. We set __signature__ to the base trainer's signature so
     every training arg is forwarded, while still accepting them via **kwargs.
"""
import inspect
from typing import Any, Optional

from brax import envs
from brax.training.agents.p3o.train import train as _base_train
from brax.envs.wrappers.threshold_budget import ThresholdBudgetWrapper
from brax.envs.wrappers.pad_obs import PadObsWrapper
import os as _os


def _apply_budget(env: envs.Env, threshold: float) -> envs.Env:
    # PAD_OBS_TO=<n> zero-pads the base obs BEFORE the budget channels, so tasks
    # with different widths (Goal 62 vs Push 68) align dim-for-dim for EWC.
    _pad = _os.environ.get('PAD_OBS_TO', '')
    if _pad:
        env = PadObsWrapper(env, int(_pad))
        print(f'[p3o_budget] PadObsWrapper -> base obs padded to {_pad}')
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
