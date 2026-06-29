"""Current estimation module for FOC control."""

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp

from .motor import Motor
from .jax_utils import register_dataclass

@register_dataclass
@dataclass
class CurrentEstimatorState:
    """State for the current estimator."""
    i_d: float  # Estimated d-axis current
    i_q: float  # Estimated q-axis current


@register_dataclass
@dataclass
class CurrentEstimator:
    """Stateful current estimator with feedforward model and back-EMF compensation.
    
    This estimator combines low-pass filtered current measurements with a feedforward
    model of the motor's electrical dynamics to provide smooth and accurate current
    estimates.
    """
    tau: float  # Time constant half life in units of steps
    feedforward: float = 1.0

    @property
    def alpha(self) -> float:
        """Calculate alpha for low-pass filter (first-order exponential smoothing)."""
        return 1 / (self.tau + 1)

    def init(self) -> CurrentEstimatorState:
        """Initialize the estimator state with zero current."""
        return CurrentEstimatorState(
            i_d=0.0,
            i_q=0.0
        )
    
    def update(
        self,
        state: CurrentEstimatorState,
        motor: Motor,
        i_d_meas: float,
        i_q_meas: float,
        v_d: float,
        v_q: float,
        omega_mech: float,
        dt: float
    ) -> Tuple[CurrentEstimatorState, float, float]:
        """Update the current estimate using measurement and feedforward model.
        
        Args:
            state: Current estimator state
            i_d_meas: Measured d-axis current (A)
            i_q_meas: Measured q-axis current (A)
            v_d: Applied d-axis voltage (V)
            v_q: Applied q-axis voltage (V)
            omega_mech: mechanical angular velocity (rad/s)
            
        Returns:
            Tuple of (new_state, i_d_est, i_q_est)
        """
        # Calculate alpha for low-pass filter (first-order exponential smoothing)
        alpha = self.alpha
        # Low-pass filter the measurements
        i_d_filtered = (1 - alpha) * state.i_d + alpha * i_d_meas
        i_q_filtered = (1 - alpha) * state.i_q + alpha * i_q_meas

        # expected changes in currents based on motor model dynamics
        di_dt_d, di_dt_q = motor.current_feedforward(v_d, v_q, i_d_filtered, i_q_filtered, omega_mech)

        # Update estimates with feedforward
        i_d_est = i_d_filtered + di_dt_d * dt * self.feedforward
        i_q_est = i_q_filtered + di_dt_q * dt * self.feedforward
        
        # Create new state
        new_state = CurrentEstimatorState(
            i_d=i_d_est,
            i_q=i_q_est
        )
        
        return new_state, i_d_est, i_q_est
