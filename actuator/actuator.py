"""Combine motor and controller into a single object"""

from dataclasses import dataclass
from typing import Tuple

import jax

from actuator.motor import MotorState, Motor
from actuator.controller import FOCController, FOCControllerState
from actuator.jax_utils import register_dataclass
from actuator.scheduling import Scheduler


@register_dataclass
@dataclass
class ActuatorState:
	"""State of the actuator.

	Attributes:
		motor: Current state of the motor
		controller: Current state of the controller
		rng_key: Random number generator key
		time: Current simulation time
	"""
	motor: MotorState
	controller: FOCControllerState
	rng_key: jax.random.PRNGKey


@register_dataclass
@dataclass
class Actuator:
	"""Actuator simulation.

	Attributes:
		motor: The motor model to simulate
		controller: The controller to use
		dt: Time step for the simulation (seconds)
		rng_key: Random number generator key
	"""
	motor: Motor
	controller: FOCController
	scheduler: Scheduler
	dt: float = 1e-4

	def init_state(self, rng_key: jax.random.PRNGKey, initial_position: float = 0.0) -> ActuatorState:
		"""Initialize the simulation state.

		Args:
			initial_position: Initial position of the motor (radians)

		Returns:
			Initialized simulation state
		"""
		motor_state = MotorState(position=initial_position)
		controller_state = self.controller.init_state(initial_position=initial_position)

		return ActuatorState(
			motor=motor_state,
			controller=controller_state,
			rng_key=rng_key,
		)

	def step(
			self,
			state: ActuatorState,
			targets: float,
			weights: float,
			load_torque: float,
	) -> Tuple[ActuatorState, Tuple[float, float]]:
		"""Perform one simulation step."""
		# Split RNG key for this step
		rng_key, enc_key = jax.random.split(state.rng_key)
		controller = self.scheduler.schedule(self, state, targets)

		# Run controller
		new_controller_state, (vd, vq) = controller.step(
			state=state.controller,
			targets=targets,
			weights=weights,
			motor_state=state.motor,
			dt=self.dt
		)

		# Update motor state
		new_motor_state = self.motor.update(
			state=state.motor,
			voltage_d=vd,
			voltage_q=vq,
			load_torque=load_torque,
			dt=self.dt
		)

		# Create new simulation state
		new_state = ActuatorState(
			motor=new_motor_state,
			controller=new_controller_state,
			rng_key=rng_key,
		)

		outputs = vq, vd

		return new_state, outputs

