"""Threshold-conditioned budget observation wrapper.

This is "Saute Piece 1" plus an OPTIONAL sink state: it appends a depleting
safety-budget channel and a static threshold channel to the observation. By
default it does NOT reshape the reward and does NOT absorb on depletion -- the
constraint is left entirely to the PID Lagrangian (PPO-PID) or the penalizer
(P3O). The base env's `cost` signal is preserved untouched.

Two appended observation dims (in this order):
    [..., z_budget, threshold_norm]

Set BUDGET_SINK=1 (env var) to enable the Saute sink: once the episode's budget
hits 0, reward is zeroed for the REST of that episode. Cost is never modified.

IMPORTANT (evaluation): the sink shapes the reward the agent optimizes. To read a
policy's TRUE task reward, evaluate it with BUDGET_SINK unset/0 -- i.e. train with
the sink on, eval with the sink off. Cost is unaffected either way.

NOTE: assumes flat-array observations (like SauteWrapper). Not for the vision/dict
obs path.
"""
import os

import jax.numpy as jnp
from brax.envs.base import Env, Wrapper, State, ObservationSize

# ============================ EDIT THESE PER EXPERIMENT ============================
# How to normalize the remaining-budget channel z_t:
#   "by_threshold" -> z = b / d      (phase-relative: always starts at 1.0)
#   "by_const"     -> z = b / CONST  (absolute scale: starts at d/CONST)
BUDGET_NORM = os.environ.get("BUDGET_NORM", "by_threshold")  # per-process; no file race

# Fixed constant for the threshold channel (and for z when BUDGET_NORM="by_const").
THRESHOLD_NORM_CONST = 25.0

# Budget depletion rule:
#   None  -> linear:      b_{t+1} = b_t - c_t
#   float -> saute-style: b_{t+1} = (b_t - c_t) / gamma
BUDGET_GAMMA = None

# Clip range applied to the *observation* value of z only (raw budget clipped >=0).
CLIP_BUDGET_OBS = (0.0, 1.5)
# ==================================================================================

# Saute sink state, toggled by env var so it can't silently leak into normal runs.
#   export BUDGET_SINK=1   -> sink ON  (reward zeroed after depletion)
#   unset / BUDGET_SINK=0  -> sink OFF (default; existing behavior, reward untouched)
BUDGET_SINK = os.environ.get("BUDGET_SINK", "0") == "1"

# Graded reshaping (Saute-INSPIRED, not Saute-exact). Canonical Saute applies a
# CONSTANT unsafe-state reward once the budget is exhausted. Here we instead subtract
# a penalty PROPORTIONAL to overspend, but CAPPED, so it provides a continuous
# conditioning gradient near the boundary WITHOUT an unbounded tail that would
# destabilize training or over-suppress reward-seeking (Option A: bounded graded).
#   export BUDGET_RESHAPE=1        -> graded penalty ON
#   export RESHAPE_SCALE=<float>   -> penalty weight (default 10.0)
#   export RESHAPE_CAP=<float>     -> max per-step penalty (default 30.0 ~ reward scale)
# penalty = min(RESHAPE_CAP, RESHAPE_SCALE * overspend_fraction)
# If both BUDGET_SINK and BUDGET_RESHAPE are set, RESHAPE takes precedence.
BUDGET_RESHAPE = os.environ.get("BUDGET_RESHAPE", "0") == "1"
RESHAPE_SCALE = float(os.environ.get("RESHAPE_SCALE", "10.0"))
RESHAPE_CAP = float(os.environ.get("RESHAPE_CAP", "30.0"))


class ThresholdBudgetWrapper(Wrapper):
    def __init__(
            self,
            env: Env,
            threshold: float = THRESHOLD_NORM_CONST,
            budget_norm: str = BUDGET_NORM,
            threshold_norm_const: float = THRESHOLD_NORM_CONST,
            budget_gamma=BUDGET_GAMMA,
            clip_budget_obs=CLIP_BUDGET_OBS,
    ):
        super().__init__(env)
        self._d = float(threshold)                       # episodic budget = safety_bound
        self._budget_norm = budget_norm
        self._const = float(threshold_norm_const)
        self._gamma = budget_gamma
        self._clip_lo, self._clip_hi = clip_budget_obs
        self._threshold_obs = self._d / self._const      # static threshold channel

    @property
    def observation_size(self) -> ObservationSize:
        return self.env.observation_size + 2

    def _norm_budget(self, b: jnp.ndarray) -> jnp.ndarray:
        denom = self._d if self._budget_norm == "by_threshold" else self._const
        z = b / denom
        return jnp.clip(z, self._clip_lo, self._clip_hi)

    def _augment_obs(self, obs: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        z = self._norm_budget(b)
        thr = jnp.ones_like(z) * self._threshold_obs
        return jnp.concatenate(
            [obs, jnp.expand_dims(z, -1), jnp.expand_dims(thr, -1)], axis=-1
        )

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        info = state.info.copy()
        if 'cost' not in info:
            info['cost'] = jnp.zeros_like(state.reward)
        b = jnp.ones_like(state.reward) * self._d
        info['tb_budget'] = b
        info['tb_depleted'] = jnp.zeros_like(b)          # sink latch, init cleared
        obs = self._augment_obs(state.obs, b)
        return state.replace(obs=obs, info=info)

    def step(self, state: State, action: jnp.ndarray) -> State:
        next_state = self.env.step(state, action)
        info = next_state.info.copy()

        cost = info.get('cost', jnp.zeros_like(next_state.reward))
        b_prev = state.info.get('tb_budget', jnp.ones_like(next_state.reward) * self._d)

        if self._gamma is None:
            b_next = b_prev - cost
        else:
            b_next = (b_prev - cost) / self._gamma
        b_next = jnp.clip(b_next, 0.0, self._d * 3.0)

        # Reset budget at the true episode boundary (AutoReset zeroes info['steps']).
        steps = state.info.get('steps', None)
        if steps is not None:
            b_next = jnp.where(steps == 0, jnp.ones_like(b_next) * self._d, b_next)

        # Depletion treatment. Priority: graded reshape > hard sink > none.
        if BUDGET_RESHAPE:
            # Budget is clipped to >=0 for the obs, but we need the RAW overspend to
            # know how far past the bound we went. Recompute the unclipped budget.
            raw_b = (b_prev - cost) if self._gamma is None else (b_prev - cost) / self._gamma
            if steps is not None:
                raw_b = jnp.where(steps == 0, jnp.ones_like(raw_b) * self._d, raw_b)
            # overspend fraction relative to the episode budget d (0 while safe, grows as we go negative)
            overspend = jnp.maximum(0.0, -raw_b) / self._d
            penalty = jnp.minimum(RESHAPE_CAP, RESHAPE_SCALE * overspend)   # bounded (Option A)
            info['tb_depleted'] = (raw_b <= 0.0).astype(b_next.dtype)   # for logging/consistency
            reward = next_state.reward - penalty
        # Saute sink: once budget hits 0, zero reward for the rest of the episode.
        elif BUDGET_SINK:
            depleted = state.info.get('tb_depleted', jnp.zeros_like(b_next))
            depleted = jnp.maximum(depleted, (b_next <= 0.0).astype(b_next.dtype))
            if steps is not None:                        # clear latch at episode start
                depleted = jnp.where(steps == 0, jnp.zeros_like(depleted), depleted)
            info['tb_depleted'] = depleted
            reward = next_state.reward * (1.0 - depleted)
        else:
            info['tb_depleted'] = state.info.get('tb_depleted', jnp.zeros_like(b_next))
            reward = next_state.reward

        info['tb_budget'] = b_next
        obs = self._augment_obs(next_state.obs, b_next)
        return next_state.replace(obs=obs, reward=reward, info=info)
