"""
Exponential Moving Average (EMA) Filter module.

This module provides a simple EMA filter that can be used for smoothing signals.
"""

from dataclasses import dataclass
from typing import Generic, TypeVar

import jax.numpy as jnp

from .jax_utils import register_dataclass

T = TypeVar('T', float, jnp.ndarray)

@register_dataclass
@dataclass
class EMAFilterState(Generic[T]):
    """State of the EMA filter.
    
    Attributes:
        value: Current filtered value
    """
    value: T


@register_dataclass
@dataclass
class EMAFilter(Generic[T]):
    """Exponential Moving Average (EMA) Filter.
    
    This filter applies an exponential moving average to smooth a signal.
    The smoothing factor alpha determines how much weight to give to new measurements.
    Smaller values of alpha result in more smoothing but slower response to changes.
    """
    
    def __init__(self, tau: float = 1., initial_value: T = 0.0):
        """Initialize the EMA filter.
        
        Args:
            tau: smoothing half-life in units of steps
                tau = 0 means no smoothing
            initial_value: Initial value for the filter.
        """
        self.tau = tau
        self.initial_value = initial_value
    
    def init(self) -> EMAFilterState[T]:
        """Initialize the filter state with the initial value."""
        return EMAFilterState(value=self.initial_value)
    
    def update(self, state: EMAFilterState[T], new_value: T) -> tuple[EMAFilterState[T], T]:
        """Update the filter with a new measurement.
        
        Args:
            state: Current filter state
            new_value: New measurement value
            
        Returns:
            Tuple of (new_state, filtered_value)
        """
        alpha = 1 / (self.tau + 1)

        # Apply EMA: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        filtered_value = alpha * new_value + (1 - alpha) * state.value
        return EMAFilterState(value=filtered_value), filtered_value
    
    def reset(self) -> EMAFilterState[T]:
        """Reset the filter to its initial state."""
        return self.init()
