"""
Controller module for PMSM (Permanent Magnet Synchronous Motor) control.

This module implements a Field-Oriented Control (FOC) strategy with cascaded
PI controllers for current and velocity control.
"""

from dataclasses import dataclass
from typing import Tuple, Any, TypeVar, Type, Dict

import jax
import jax.numpy as jnp
from jax._src.prng import PRNGKeyArray as KeyArray

from .jax_utils import register_dataclass
from .utils import angle_difference_wrap, lerp

from .encoder import Encoder
from .current_sensor import CurrentSensor
from .observer import EMAPositionVelocityObserver, EMAPositionVelocityObserverState
from .current_estimator import CurrentEstimator, CurrentEstimatorState
from .motor import MotorState, Motor
from .pi_controller import PIController, PIControllerState



@register_dataclass
@dataclass
class FOCControllerState:
    """State of the FOC controller.
    """
    observer: EMAPositionVelocityObserverState
    current_estimator: CurrentEstimatorState
    position_ctrl: PIControllerState
    velocity_ctrl: PIControllerState
    iq_ctrl: PIControllerState
    id_ctrl: PIControllerState
    rng_key: KeyArray
    
    @classmethod
    def create(
        cls,
        observer: EMAPositionVelocityObserver,
        current_estimator: CurrentEstimator,
        position_ctrl: PIController,
        velocity_ctrl: PIController,
        iq_ctrl: PIController,
        id_ctrl: PIController,
        rng_key: KeyArray,
        initial_position: float = 0.0
    ) -> 'FOCControllerState':
        """Create a new FOC controller state with initialized components.
        """
        return cls(
            observer=observer.reset(initial_position),
            position_ctrl=position_ctrl.init_state(),
            velocity_ctrl=velocity_ctrl.init_state(),
            iq_ctrl=iq_ctrl.init_state(),
            id_ctrl=id_ctrl.init_state(),
            current_estimator=current_estimator.init(),
            rng_key=rng_key
        )


@register_dataclass
@dataclass
class FOCController:
    """Field-Oriented Control (FOC) for PMSM with encoder feedback.
    """
    # this holds the motor parameters used for the internal model for ff computations and such; disjoint from physical motor
    motor: Motor

    # estimator: maps measurements to state estimate
    encoder: Encoder
    current_sensor: CurrentSensor
    observer: EMAPositionVelocityObserver
    current_estimator: CurrentEstimator

    # controller; maps state estimate to voltage commands
    position_ctrl: PIController
    velocity_ctrl: PIController
    iq_ctrl: PIController
    id_ctrl: PIController
    voltage_feedforward: float    # 1 means full feedforward, 0 means no feedforward

    max_current: float  # A
    max_voltage: float  # V

    @classmethod
    def create_default(
        cls, 
        motor: Motor,
        encoder: Encoder,
        max_current: float = 100,
        max_voltage: float = 75,
        max_velocity: float = 100.0,  # rad/s
    ) -> 'FOCController':
        """Create a controller with default tuning for the given motor.
        
        Note: Current and voltage limits are properties of the controller, not the motor.
        The motor's electrical parameters (resistance, inductance, etc.) are physical
        properties, while current/voltage limits are determined by the controller's
        power electronics and cooling system.
        
        Returns:
            Configured FOC controller
        """
        observer = EMAPositionVelocityObserver()

        # Default controller gains; conservative and motor-agnostic, intended as a
        # starting point for optimization rather than a tuned configuration.
        position_bandwidth = 5.0  # rad/s
        velocity_bandwidth = 5.0  # rad/s
        current_bandwidth = 5.0   # rad/s

        # position controller (inner loop)
        position_kp = 1.1
        position_ki = position_kp * position_bandwidth

        # Velocity controller (outer loop)
        velocity_kp = 1.1
        velocity_ki = velocity_kp * velocity_bandwidth
        
        # Current controllers (inner loops)
        current_kp = 1.1
        current_ki = current_kp * current_bandwidth
        
        return cls(
            motor=motor,
            current_sensor=CurrentSensor(noise_std=0.1),
            position_ctrl=PIController(
                kp=position_kp,
                ki=position_ki,
                output_min=-max_velocity,
                output_max=max_velocity
            ),
            velocity_ctrl=PIController(
                kp=velocity_kp,
                ki=velocity_ki,
                output_min=-max_current,
                output_max=max_current
            ),
            iq_ctrl=PIController(
                kp=current_kp,
                ki=current_ki,
                output_min=-max_voltage,
                output_max=max_voltage
            ),
            id_ctrl=PIController(
                kp=current_kp,
                ki=current_ki,
                output_min=-max_voltage,
                output_max=max_voltage
            ),
            encoder=encoder,
            observer=observer,
            current_estimator=CurrentEstimator(tau=0.01),
            max_current=max_current,
            max_voltage=max_voltage,
            voltage_feedforward=1.0,
        )

    def reset(self, initial_position: float = 0.0) -> FOCControllerState:
        """Reset all controller states to their initial values.

        Args:
            initial_position: Initial position in radians

        Returns:
            A new controller state with all components reset
        """
        return FOCControllerState.create(
            position_ctrl=self.position_ctrl,
            velocity_ctrl=self.velocity_ctrl,
            iq_ctrl=self.iq_ctrl,
            id_ctrl=self.id_ctrl,
            observer=self.observer,
            current_estimator=self.current_estimator,
            initial_position=initial_position,
            rng_key=jax.random.PRNGKey(42),
        )

    def init_state(self, initial_position: float = 0.0) -> FOCControllerState:
        """Initialize the controller state.
        
        Args:
            initial_position: Initial position in radians
            
        Returns:
            A new controller state with all components initialized
        """
        return self.reset(initial_position=initial_position)

    def step_voltage(self,
        state: FOCControllerState,
        estimated_velocity: float,
        i_d_est: float,
        i_q_est: float,
        iq_target: float,
        dt: float):

        # Current control (inner loops)
        id_target = 0  # Target id = 0 for maximum torque per amp; assuming non-salient motor
        iq_error = iq_target - i_q_est
        id_error = id_target - i_d_est

        # Voltage feedforward (di/dt terms left at zero; steady-state feedforward only)
        vd_ff, vq_ff = self.motor.voltage_feedforward(i_d_est, i_q_est, 0, 0, estimated_velocity)

        # PI control + feedforward.
        # NOTE: vd and vq are limited separately below, whereas the physically
        #  relevant constraint is on current magnitude. This is acceptable while
        #  id_target = 0, but would need revisiting for field weakening.
        vd, id_ctrl_state = self.id_ctrl.step(state.id_ctrl, id_error, dt)
        vd += vd_ff * self.voltage_feedforward

        vq, iq_ctrl_state = self.iq_ctrl.step(state.iq_ctrl, iq_error, dt)
        vq += vq_ff * self.voltage_feedforward

        # Voltage limiting using jnp.clip for cleaner code
        v_abs = jnp.sqrt(vd ** 2 + vq ** 2)
        scale = jnp.minimum(1.0, self.max_voltage / (v_abs + 1e-8))  # Add small epsilon to avoid division by zero
        vd = vd * scale
        vq = vq * scale
        return vd, vq, id_ctrl_state, iq_ctrl_state

    def step(
        self,
        state: FOCControllerState,
        targets: jnp.ndarray,
        weights: jnp.ndarray,
        motor_state: MotorState,
        dt: float
    ) -> Tuple[FOCControllerState, Tuple[float, float]]:
        """Compute the control output for the next time step.
        
        Args:
            state: Current controller state
            targets: torque (Nm), velocity (rad/s), and position (rad) targets
            weights: weight for each target
            motor_state: Current motor state (includes true position and currents)
            dt: Time step in seconds
            
        Returns:
            Tuple of ((voltage_d, voltage_q), new_state) in the rotor reference frame
        """
        key, mkey, ckey = jax.random.split(state.rng_key, 3)

        target_t, target_v, target_p = targets
        weight_t, weight_v, weight_p = weights

        # Get encoder measurement
        measured_position = self.encoder.measure(motor_state, key=mkey)

        measured_current_d, measured_current_q = self.current_sensor.measure(motor_state, ckey)
        # NOTE: currents are measured in the true rotor dq frame. To simulate
        #  ABC-frame conversions, we would rotate the qd frame by the frame error
        #  (estimated_position - motor_state.position) here.


        observer_state, estimated_position, estimated_velocity = self.observer.update(state.observer, measured_position, dt)
        frame_error = angle_difference_wrap(motor_state.position - estimated_position)


        # position control
        target_velocity, position_ctrl_state = self.position_ctrl.step(
            state.position_ctrl,
            angle_difference_wrap(target_p - estimated_position),
            dt
        )

        # Velocity control (outer loop
        velocity_error = lerp(target_velocity, target_v, weight_v) - estimated_velocity
        current_target, velocity_ctrl_state = self.velocity_ctrl.step(
            state.velocity_ctrl, velocity_error, dt)

        # Current control
        torque_target = lerp(current_target * self.motor.Kt, target_t, weight_t)
        iq_target = torque_target / self.motor.Kt
        # reapply current limit after torque command lerping
        iq_target = jnp.clip(iq_target, -self.max_current, self.max_current)


        # compute ff voltages with raw measured current
        vd, vq, _, _ = self.step_voltage(
            state,
            estimated_velocity,
            measured_current_d, measured_current_q,
            iq_target, dt)

        # Update current estimates with feedforward model
        current_estimator_state, i_d_est, i_q_est = self.current_estimator.update(
            state=state.current_estimator,
            motor=self.motor,
            i_d_meas=measured_current_d,
            i_q_meas=measured_current_q,
            v_d=vd,
            v_q=vq,
            omega_mech=estimated_velocity,
            dt=dt
        )

        # compute ff voltages with estimated current
        vd, vq, id_ctrl_state, iq_ctrl_state = self.step_voltage(
            state,
            estimated_velocity,
            i_d_est, i_q_est,
            iq_target, dt
        )
        
        # Create a new state with updated fields
        new_state = FOCControllerState(
            observer=observer_state,
            position_ctrl=position_ctrl_state,
            velocity_ctrl=velocity_ctrl_state,
            iq_ctrl=iq_ctrl_state,
            id_ctrl=id_ctrl_state,
            # velocity_estimator=velocity_estimator_state,
            current_estimator=current_estimator_state,
            rng_key=key,
        )
        # NOTE: voltages are returned in the true rotor dq frame. If we ever want
        #  to simulate ABC-frame conversions (i.e. the controller acting in its
        #  *estimated* frame), we would rotate the qd frame by the frame error
        #  (estimated_position - motor_state.position) here.
        return new_state, (vd, vq)

