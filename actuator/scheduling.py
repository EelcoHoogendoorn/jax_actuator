"""
Gains scheduling functionality
"""

from dataclasses import dataclass

from jax import numpy as jnp

from actuator.controller import FOCController
from actuator.jax_utils import register_dataclass, lerp_pytree
from actuator.utils import angle_difference_wrap


@register_dataclass
@dataclass
class Scheduler:
	"""dummy scheduler"""
	c0: FOCController
	def schedule(self, actuator, state, targets):
		return self.c0


@register_dataclass
@dataclass
class TorqueScheduler:
	c0: FOCController
	c1: FOCController
	deadband: float = 0 	# 0.005
	gain: float = 20		# radian error at which c1 maxes out
	def schedule(self, actuator, state, targets):
		"""quadratic ramp between c0 and c1"""
		weight_p = jnp.abs(angle_difference_wrap(targets[2] - state.controller.observer.position))
		weight_i = jnp.sqrt(state.controller.current_estimator.i_q**2+state.controller.current_estimator.i_q**2) / 50
		weight = jnp.clip((weight_p - self.deadband) * self.gain + weight_i, 0, 1)
		return lerp_pytree(self.c0, self.c1, weight)


@register_dataclass
@dataclass
class TractionScheduler:
	c0: FOCController
	c1: FOCController
	deadband: float = 30	# starting torque of ramp
	gain: float = 30		# length of ramp
	def schedule(self, actuator, state, targets):
		torque = actuator.motor.torque(state.motor)
		weightt = jnp.clip((jnp.abs(torque) - self.deadband) / self.gain, 0, 1)
		return lerp_pytree(self.c0, self.c1, weightt)
