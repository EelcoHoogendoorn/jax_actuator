"""
Encoder module for position sensing.

This module provides functionality for simulating incremental encoders with configurable
resolution and noise. The encoder is stateless and provides position measurements
with optional quantization and noise effects.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random

from actuator.jax_utils import register_dataclass
from actuator.motor import Motor, MotorState

# Type alias for JAX PRNG key
KeyArray = jax.Array


@register_dataclass
@dataclass
class Encoder:
    """Stateless absolute encoder model with configurable resolution and noise.

    This class simulates an absolute encoder with the following features:
    - Position quantization based on encoder resolution
    - Additive Gaussian noise on position measurements
    - No internal state (stateless operation)

    Attributes:
        resolution: Number of counts per revolution (CPR)
        noise_std: Standard deviation of the position measurement noise (in counts)
        phase_offset: Mechanical offset of the encoder (in radians)
    """
    resolution: int = 2000  # Counts per revolution (CPR)
    noise_std: float = 0.1  # Counts
    phase_offset: float = 0.0  # Radians

    def init_state(self):
        return ()

    @property
    def counts_to_rad(self) -> float:
        return 2 * jnp.pi / self.resolution
    @property
    def rad_to_counts(self) -> float:
        return self.resolution / (2 * jnp.pi)

    def measure(
        self,
        state: MotorState,
        key: KeyArray
    ) -> float:
        """Get a position measurement.

        Args:
            true_position: True position in radians
            key: JAX PRNG key for noise generation (optional)

        Returns:
            Measured position in radians (with noise and quantization)
        """
        # Apply phase offset
        position = state.position + self.phase_offset

        # Convert true position to counts
        true_counts = position * self.rad_to_counts


        # Add noise if key is provided
        noise = random.normal(key) * self.noise_std
        noisy_counts = true_counts + noise

        # Add quantization
        quantized_counts = jnp.round(noisy_counts)

        # Convert back to radians and remove phase offset
        measured_position = quantized_counts * self.counts_to_rad - self.phase_offset

        return measured_position
