
"""Pad a shorter observation to a target width with trailing zeros.



CRAX Goal's observation is EXACTLY the first 62 dims of CRAX Push's:

    [0:12]  accel/vel/gyro/mag     [12:28] goal lidar     [28:44] hazard lidar

    [44:46] goal compass           [46:62] hazard compasses

Push then appends 6 block-specific terms (agent->block compass, block->goal

compass, and the two normalised distances). Padding Goal to Push's width with

trailing zeros therefore aligns the two tasks dimension-for-dimension, which is

what EWC's (theta - theta*)^2 requires across a task boundary.



The zeros are semantically correct, not arbitrary: in Goal there IS no block, so

"distance to block" being zero is the right encoding. A Goal-trained policy will

also have ~0 Fisher on the weights reading dims 62-67, so EWC will not protect

them and later Push training is free to use them.

"""

import jax.numpy as jnp

from brax.envs.base import Env, Wrapper, State, ObservationSize





class PadObsWrapper(Wrapper):

    def __init__(self, env: Env, target_size: int):

        super().__init__(env)

        base = env.observation_size

        if isinstance(base, (tuple, list)):

            base = base[0]

        self._base = int(base)

        self._target = int(target_size)

        self._pad = self._target - self._base

        if self._pad < 0:

            raise ValueError(f"target_size {target_size} < env obs {self._base}")



    @property

    def observation_size(self) -> ObservationSize:

        return self._target



    def _pad_obs(self, obs: jnp.ndarray) -> jnp.ndarray:

        if self._pad == 0:

            return obs

        z = jnp.zeros(obs.shape[:-1] + (self._pad,), dtype=obs.dtype)

        return jnp.concatenate([obs, z], axis=-1)



    def reset(self, rng: jnp.ndarray) -> State:

        s = self.env.reset(rng)

        return s.replace(obs=self._pad_obs(s.obs))



    def step(self, state: State, action: jnp.ndarray) -> State:

        s = self.env.step(state, action)

        return s.replace(obs=self._pad_obs(s.obs))

