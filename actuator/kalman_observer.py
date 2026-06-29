"""Prototype kalman position+velocity observer"""


import jax
import jax.numpy as jnp
from jax import jit
from typing import NamedTuple, Tuple
import chex

from .jax_utils import register_dataclass
from dataclasses import dataclass

@register_dataclass
@dataclass
class KalmanState():
	"""State of the Kalman filter"""
	x: chex.Array  # State vector [position, velocity]
	P: chex.Array  # Covariance matrix (2x2)


@register_dataclass
@dataclass
class KalmanParams():
	"""Parameters for the Kalman filter"""
	A: chex.Array  # State transition matrix (2x2)
	B: chex.Array  # Input matrix (2x1)
	H: chex.Array  # Measurement matrix (1x2)
	Q: chex.Array  # Process noise covariance (2x2)
	R: chex.Array  # Measurement noise covariance (1x1)


def create_kalman_params(dt: float,
						 process_noise_pos: float = 1e-6,
						 process_noise_vel: float = 1e-4,
						 measurement_noise: float = 1e-4) -> KalmanParams:
	"""Create Kalman filter parameters for position-velocity system

	Args:
		dt: Sample time
		process_noise_pos: Process noise for position
		process_noise_vel: Process noise for velocity
		measurement_noise: Measurement noise for position sensor
	"""

	# State transition matrix: [pos, vel] -> [pos + vel*dt, vel]
	A = jnp.array([[1.0, dt],
				   [0.0, 1.0]])

	# Input matrix: acceleration input affects velocity
	B = jnp.array([[0.0],
				   [dt]])

	# Measurement matrix: we only measure position
	H = jnp.array([[1.0, 0.0]])

	# Process noise covariance
	Q = jnp.array([[process_noise_pos, 0.0],
				   [0.0, process_noise_vel]])

	# Measurement noise covariance (scalar -> 1x1 matrix)
	R = jnp.array([[measurement_noise]])

	return KalmanParams(A=A, B=B, H=H, Q=Q, R=R)


def init_kalman_state(initial_position: float = 0.0,
					  initial_velocity: float = 0.0,
					  initial_pos_uncertainty: float = 0.1,
					  initial_vel_uncertainty: float = 1.0) -> KalmanState:
	"""Initialize Kalman filter state"""

	x = jnp.array([initial_position, initial_velocity])
	P = jnp.array([[initial_pos_uncertainty, 0.0],
				   [0.0, initial_vel_uncertainty]])

	return KalmanState(x=x, P=P)


@jit
def kalman_predict(state: KalmanState,
				   params: KalmanParams,
				   acceleration_input: float = 0.0) -> KalmanState:
	"""Kalman filter prediction step

	Args:
		state: Current Kalman state
		params: Kalman parameters
		acceleration_input: Acceleration command/input
	"""

	# State prediction: x_pred = A @ x + B @ u
	u = jnp.array([acceleration_input])
	x_pred = params.A @ state.x + (params.B @ u).flatten()

	# Covariance prediction: P_pred = A @ P @ A^T + Q
	P_pred = params.A @ state.P @ params.A.T + params.Q

	return KalmanState(x=x_pred, P=P_pred)


@jit
def kalman_update(predicted_state: KalmanState,
				  params: KalmanParams,
				  measurement: float) -> Tuple[KalmanState, dict]:
	"""Kalman filter update step

	Args:
		predicted_state: Predicted state from prediction step
		params: Kalman parameters
		measurement: Position measurement

	Returns:
		Updated state and diagnostics dict
	"""

	# Convert scalar measurement to array
	z = jnp.array([measurement])

	# Innovation: y = z - H @ x_pred
	y = z - params.H @ predicted_state.x

	# Innovation covariance: S = H @ P_pred @ H^T + R
	S = params.H @ predicted_state.P @ params.H.T + params.R

	# Kalman gain: K = P_pred @ H^T @ inv(S)
	K = predicted_state.P @ params.H.T @ jnp.linalg.inv(S)

	# State update: x = x_pred + K @ y
	x_updated = predicted_state.x + (K @ y).flatten()

	# Covariance update: P = (I - K @ H) @ P_pred
	I = jnp.eye(2)
	P_updated = (I - K @ params.H) @ predicted_state.P

	# Diagnostics
	diagnostics = {
		'innovation': y[0],
		'innovation_covariance': S[0, 0],
		'kalman_gain': K.flatten(),
		'log_likelihood': -0.5 * (jnp.log(2 * jnp.pi * S[0, 0]) + y[0] ** 2 / S[0, 0])
	}

	updated_state = KalmanState(x=x_updated, P=P_updated)
	return updated_state, diagnostics


@jit
def kalman_step(state: KalmanState,
				params: KalmanParams,
				measurement: float,
				acceleration_input: float = 0.0) -> Tuple[KalmanState, dict]:
	"""Complete Kalman filter step (predict + update)

	Args:
		state: Current Kalman state
		params: Kalman parameters
		measurement: Position measurement
		acceleration_input: Acceleration input/command

	Returns:
		Updated state and diagnostics
	"""

	# Prediction step
	predicted_state = kalman_predict(state, params, acceleration_input)

	# Update step
	updated_state, diagnostics = kalman_update(predicted_state, params, measurement)

	return updated_state, diagnostics


# Encoder-specific utilities
def encoder_to_position(encoder_count: int,
						encoder_resolution: int) -> float:
	"""Convert encoder count to position in radians"""
	return (encoder_count * 2 * jnp.pi) / encoder_resolution


@jit
def unwrap_position_difference(current_pos: float, previous_pos: float) -> float:
	"""Unwrap position difference for continuous tracking"""
	diff = current_pos - previous_pos
	# Wrap to [-π, π]
	diff = jnp.where(diff > jnp.pi, diff - 2 * jnp.pi, diff)
	diff = jnp.where(diff < -jnp.pi, diff + 2 * jnp.pi, diff)
	return previous_pos + diff


@register_dataclass
@dataclass
class MotorObserverState:
	state: KalmanState
	previous_position: float


# Example usage class
@register_dataclass
@dataclass
class MotorObserver:
	"""Motor position-velocity observer using Kalman filter"""

	def __init__(self, dt: float, encoder_resolution: int):
		self.dt = dt
		self.encoder_resolution = encoder_resolution
		self.params = create_kalman_params(dt)

	def init_state(self):
		return MotorObserverState(state=init_kalman_state(), previous_position=0.0)
	def update(self, state, encoder_count: int, acceleration_cmd: float = 0.0) -> dict:
		"""Update observer with new encoder measurement

		Args:
			encoder_count: Raw encoder count
			acceleration_cmd: Acceleration command (optional)

		Returns:
			Dictionary with position, velocity, and diagnostics
		"""

		# Convert encoder to position
		raw_position = encoder_to_position(encoder_count, self.encoder_resolution)

		# Handle wraparound for continuous tracking
		unwrapped_position = unwrap_position_difference(raw_position, state.previous_position)

		# Run Kalman filter step
		state, diagnostics = kalman_step(
			state.state, self.params, unwrapped_position, acceleration_cmd
		)

		return MotorObserverState(state=state, previous_position=raw_position), {
			'position': (state.x[0]),
			'velocity': (state.x[1]),
			'position_uncertainty': (jnp.sqrt(state.P[0, 0])),
			'velocity_uncertainty': (jnp.sqrt(state.P[1, 1])),
			'diagnostics': diagnostics
		}


# Example usage and test
if __name__ == "__main__":
	# Create observer
	observer = MotorObserver(dt=0.001, encoder_resolution=4096)  # 1kHz, 4096 CPR

	# Simulate some measurements
	jax.random.PRNGKey(42)

	print("Time\tEncoder\tPosition\tVelocity\tPos_Std\tVel_Std")
	print("-" * 60)

	for i in range(10):
		# Simulate encoder count (ramping up)
		encoder_count = i * 100 + int(10 * jnp.sin(i * 0.1))  # Some noise

		# Update observer
		result = observer.update(encoder_count)

		print(f"{i * 0.001:.3f}\t{encoder_count}\t{result['position']:.4f}\t"
			  f"{result['velocity']:.2f}\t{result['position_uncertainty']:.4f}\t"
			  f"{result['velocity_uncertainty']:.2f}")