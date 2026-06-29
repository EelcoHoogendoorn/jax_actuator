"""
Proportional-Integral (PI) controller implementation.

This module provides a PI controller with anti-windup protection and state management.
"""

from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp

from .jax_utils import register_dataclass

@register_dataclass
@dataclass
class PIControllerState:
    """State of a PI controller.
    
    Attributes:
        integral: Current integral term
        last_error: Previous control error
        last_output: Previous controller output (for rate limiting)
    """
    integral: float = 0.0
    last_error: float = 0.0
    last_output: float = 0.0


@register_dataclass
@dataclass
class PIController:
    """Proportional-Integral (PI) controller with anti-windup.
    
    Attributes:
        kp: Proportional gain
        ki: Integral gain
        output_min: Minimum output limit
        output_max: Maximum output limit
    """
    kp: float
    ki: float
    output_min: float
    output_max: float
    kd: float = 0.0
    decay: float = 0.0  # integrator decay
    max_rate: float = 999999  # maximum rate of change (units/step)

    def init_state(self) -> PIControllerState:
        """Initialize the controller state.
        
        Returns:
            A new controller state with zero integral term and output.
        """
        return PIControllerState(integral=0.0, last_error=0.0, last_output=0.0)
    
    def step(
        self,
        state: PIControllerState,
        error: float,
        dt: float
    ) -> Tuple[float, PIControllerState]:
        """Compute the controller output for the given error and time step.
        
        Args:
            state: Current controller state
            error: Current control error
            dt: Time step in seconds
            
        Returns:
            Tuple of (output, new_state)
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        i_term = (jnp.exp(-self.decay)) * state.integral + self.ki * error * dt

        d_term = self.kd * (error - state.last_error) / dt
        
        # Calculate output with clamping
        output = p_term + i_term + d_term
        
        # Apply output limits
        output_clipped = jnp.clip(output, self.output_min, self.output_max)
        
        # Apply rate limiting
        max_delta = self.max_rate
        output_limited = jnp.clip(
            jnp.clip(
                output_clipped,
                state.last_output - max_delta,
                state.last_output + max_delta
            ),
            self.output_min,
            self.output_max
        )
        
        # Anti-windup: only integrate if not saturated
        new_integral = jnp.where(
            (output >= self.output_min) & (output <= self.output_max),
            i_term,  # Update integral if not saturated
            state.integral  # Keep current integral if saturated
        )
        
        return output_limited, PIControllerState(
            integral=new_integral,
            last_error=error,
            last_output=output_limited
        )
    
    def reset(self) -> PIControllerState:
        """Reset the controller state.
        
        Returns:
            A new controller state with zero integral term.
        """
        return self.init_state()
