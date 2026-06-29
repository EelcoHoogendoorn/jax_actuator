"""Tests for the EMA filter module."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as jnp

# Import the EMA filter module
from actuator.ema_filter import EMAFilter, EMAFilterState


def test_ema_filter_basic():
    """Test basic EMA filter functionality."""
    # Create a filter with tau=1 (alpha = 1/(tau+1) = 0.5) and initial value = 0.0
    tau = 1.0
    alpha = 1 / (tau + 1)  # 0.5
    ema = EMAFilter(tau=tau, initial_value=0.0)

    # Initialize the filter
    state = ema.init()

    # Test a step response
    values = [1.0] * 10
    filtered_values = []

    for v in values:
        state, filtered = ema.update(state, v)
        filtered_values.append(filtered)

    # Check that the filter converges to the input
    expected = 1.0 - (1.0 - alpha) ** np.arange(1, 11)
    np.testing.assert_allclose(filtered_values, expected, atol=1e-6)


def test_ema_filter_initial_value():
    """Test EMA filter with custom initial value."""
    initial_value = 5.0
    ema = EMAFilter(tau=1.0, initial_value=initial_value)
    state = ema.init()

    # First update should use the initial value (alpha=0.5)
    state, filtered = ema.update(state, 1.0)
    assert filtered == 0.5 * 1.0 + 0.5 * initial_value


def test_ema_filter_visual():
    """Visual test of the EMA filter with different alpha values."""
    # Generate a noisy signal
    t = np.linspace(0, 2*np.pi, 200)
    signal = np.sin(t) + 0.2 * np.random.normal(size=len(t))
    
    # Test different tau values (corresponding to alpha = 1/(tau+1))
    taus = [9.0, 7/3, 3/7]  # alpha ≈ 0.1, 0.3, 0.7
    filtered_signals = {}

    for tau in taus:
        alpha = 1 / (tau + 1)
        ema = EMAFilter(tau=tau, initial_value=signal[0])
        state = ema.init()

        filtered = []
        for s in signal:
            state, f = ema.update(state, s)
            filtered.append(f)

        filtered_signals[f'{alpha:.1f}'] = np.array(filtered)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, 'k-', alpha=0.3, label='Noisy Signal')
    
    for label, filtered in filtered_signals.items():
        plt.plot(t, filtered, label=f'EMA (α={label})')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('EMA Filter Response to Noisy Sinusoid')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'ema_filter_response.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    
    print(f"Visual test complete. Plot saved to {output_path}")


if __name__ == "__main__":
    print("Running basic EMA filter test...")
    test_ema_filter_basic()
    
    print("\nTesting with initial value...")
    test_ema_filter_initial_value()
    
    print("\nRunning visual test...")
    test_ema_filter_visual()
    
    print("\nAll tests passed!")
