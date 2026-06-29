from dataclasses import dataclass

from .jax_utils import register_dataclass
from .utils import angle_difference_wrap


def alpha(tau):
	alpha = 1 / (1 + tau)
	return lambda l, r: l * (1 - alpha) + r * alpha


@register_dataclass
@dataclass
class EMAPositionVelocityObserverState:
	position: float = 0
	velocity: float = 0


@register_dataclass
@dataclass
class EMAPositionVelocityObserver:
	"""Complementary position/velocity observer: feedforward velocity prediction
	combined with EMA correction from the encoder measurement."""
	tau_pos: float = 0.1
	tau_vel: float = 0.1

	def update(self, state: EMAPositionVelocityObserverState, position_measured: float, dt: float):
		# Feedforward prediction step
		position_predicted = state.position + state.velocity * dt

		# Handle position wrapping using the wrapped difference for position update
		position_diff = angle_difference_wrap(position_measured - position_predicted)
		position_est = position_predicted + alpha(self.tau_pos)(0, position_diff)

		# Velocity update using wrapped position difference
		wrapped_position_diff = angle_difference_wrap(position_measured - state.position)
		velocity_correction = wrapped_position_diff / dt
		velocity_est = alpha(self.tau_vel)(state.velocity, velocity_correction)

		return EMAPositionVelocityObserverState(position_est, velocity_est), position_est, velocity_est

	def reset(self, position=0.0, velocity=0.0):
		return EMAPositionVelocityObserverState(position, velocity)