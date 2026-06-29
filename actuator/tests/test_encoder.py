"""Tests for the encoder module."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as jnp

# Import the encoder module
from actuator.encoder import Encoder
from actuator.motor import MotorState


def test_encoder_noise():
    """Test that the encoder adds noise when a key is provided."""
    # Set a fixed random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create an encoder with noise
    encoder = Encoder(
        resolution=1000,  # 1000 counts/rev
        noise_std=0.5     # 0.5 counts noise
    )
    
    # Test position measurements
    positions = np.linspace(0, 2*np.pi, 100)
    measured_positions = []
    
    for pos in positions:
        key, subkey = jax.random.split(key)
        meas_pos = encoder.measure(
            state=MotorState(position=pos),
            key=subkey
        )
        measured_positions.append(meas_pos)
    
    # Check that the measurements are not exactly equal to the quantized positions
    # due to noise
    expected_counts = np.round(positions * encoder.resolution / (2*np.pi))
    expected_positions = expected_counts * (2*np.pi / encoder.resolution)
    
    # There should be some difference due to noise
    assert not np.allclose(measured_positions, expected_positions, atol=1e-6)
    # But the difference should be small
    assert np.allclose(measured_positions, expected_positions, atol=0.1)


def test_encoder_phase_offset():
    """Test that the phase offset is correctly applied."""
    # Create an encoder with a phase offset
    phase_offset = 0.5  # radians
    encoder = Encoder(
        resolution=1000,
        noise_std=0.0,
        phase_offset=phase_offset
    )

    # Test position measurements
    key = jax.random.PRNGKey(0)
    positions = np.linspace(0, 2*np.pi, 10)

    for pos in positions:
        key, subkey = jax.random.split(key)
        # The phase offset should be applied and then removed
        measured = encoder.measure(state=MotorState(position=pos), key=subkey)
        expected = (jnp.round((pos + phase_offset) * encoder.resolution / (2*jnp.pi)) * 
                   (2*jnp.pi / encoder.resolution) - phase_offset)
        np.testing.assert_allclose(measured, expected, atol=1e-6)


def test_encoder_visual():
    """Visual test of the encoder with and without noise."""
    # Set a fixed random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create two encoders: one with noise, one without
    encoder_clean = Encoder(
        resolution=10,
        noise_std=0.0
    )
    
    encoder_noisy = Encoder(
        resolution=10,
        noise_std=0.5
    )
    
    # Generate a smooth position profile (sine wave)
    t = np.linspace(0, 1.0, 1000)  # 1 second simulation
    true_positions = 2 * np.pi * np.sin(2 * np.pi * 2 * t)  # 2Hz sine wave
    
    # Simulate both encoders
    clean_meas = []
    noisy_meas = []
    
    for i, (time, pos) in enumerate(zip(t, true_positions)):
        # Get clean measurement
        key, subkey = jax.random.split(key)
        clean_pos = encoder_clean.measure(
            state=MotorState(position=pos),
            key=subkey
        )
        
        # Get noisy measurement
        noisy_pos = encoder_noisy.measure(
            state=MotorState(position=pos),
            key=subkey
        )
        
        clean_meas.append(clean_pos)
        noisy_meas.append(noisy_pos)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, true_positions, 'k-', label='True Position')
    plt.plot(t, clean_meas, 'b-', label='Clean Encoder')
    plt.plot(t, noisy_meas, 'r.', label='Noisy Encoder', markersize=2)
    plt.ylabel('Position (rad)')
    plt.title('Encoder Position Measurements')
    plt.legend()
    plt.grid(True)
    
    # Plot error
    plt.subplot(2, 1, 2)
    plt.plot(t, np.array(clean_meas) - true_positions, 'b-', label='Clean Error')
    plt.plot(t, np.array(noisy_meas) - true_positions, 'r-', label='Noisy Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad)')
    plt.title('Measurement Error')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'encoder_visualization.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    
    print(f"Visual test complete. Plot saved to {output_path}")


if __name__ == "__main__":
    print("Running basic encoder test...")
    test_encoder_basic()
    
    print("\nRunning noise test...")
    test_encoder_noise()
    
    print("\nRunning phase offset test...")
    test_encoder_phase_offset()
    
    print("\nRunning visual test...")
    test_encoder_visual()
    
    print("\nAll tests passed!")
