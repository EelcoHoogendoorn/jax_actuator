from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random

from actuator.jax_utils import register_dataclass
from actuator.motor import MotorState

# Type alias for JAX PRNG key
KeyArray = jax.Array


@register_dataclass
@dataclass
class CurrentSensor:
	noise_std: float = 0.1

	def measure(self, state: MotorState, rng: KeyArray) -> float:
		nd, nq =jax.random.normal(rng, shape=(2,)) * self.noise_std
		return state.current_d + nd, state.current_q + nq
