"""Tests for the PI controller module."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

from actuator.pi_controller import PIController, PIControllerState


def test_pi_controller_step():
    """Test basic PI controller step functionality."""
    # Create a PI controller with simple gains
    controller = PIController(
        kp=1.0,
        ki=0.1,
        output_min=-10.0,
        output_max=10.0
    )
    
    # Initial state
    state = controller.init_state()
    
    # Test proportional response
    output, state = controller.step(state, error=1.0, dt=0.1)
    # First step: output = kp * error + ki * error * dt = 1.0 * 1.0 + 0.1 * 1.0 * 0.1 = 1.01
    assert abs(output - 1.01) < 1e-6
    
    # Test integral accumulation
    output, state = controller.step(state, error=1.0, dt=0.1)
    # Second step: output = kp * error + ki * (integral + error * dt) = 1.0 * 1.0 + 0.1 * (0.1 + 1.0 * 0.1) = 1.02
    assert abs(output - 1.02) < 1e-6
    
    # Test output saturation
    controller = PIController(
        kp=100.0,  # Large gain to ensure saturation
        ki=10.0,
        output_min=-1.0,
        output_max=1.0
    )
    state = controller.init_state()
    output, state = controller.step(state, error=1.0, dt=0.1)
    assert abs(output - 1.0) < 1e-6  # Should saturate at 1.0


def test_pi_controller_anti_windup():
    """Test PI controller anti-windup functionality."""
    # Create a PI controller that will saturate
    controller = PIController(
        kp=2.0,
        ki=1.0,
        output_min=-1.0,
        output_max=1.0
    )
    state = controller.init_state()
    
    # First step - should saturate
    output1, state1 = controller.step(state, error=1.0, dt=0.1)
    assert abs(output1 - 1.0) < 1e-6  # Should saturate at 1.0
    
    # Second step with same error - integral should not accumulate
    output2, state2 = controller.step(state1, error=1.0, dt=0.1)
    assert abs(output2 - 1.0) < 1e-6  # Still saturated
    assert state2.integral == state1.integral  # Integral should not change
    
    # Step with negative error - should come out of saturation
    output3, state3 = controller.step(state2, error=-2.0, dt=0.1)
    assert output3 < 1.0  # No longer saturated


def test_pi_controller_tracking():
    """Test PI controller tracking performance."""
    # Create a well-tuned PI controller
    controller = PIController(
        kp=2.0,
        ki=5.0,
        output_min=-10.0,
        output_max=10.0
    )
    state = controller.init_state()
    
    # Simulate step response
    dt = 0.01
    target = 1.0
    output = 0.0
    outputs = []
    
    for _ in range(100):  # 1 second simulation
        error = target - output
        control, state = controller.step(state, error, dt)
        # Simple first-order plant model
        output += (control - output) * dt * 5.0
        outputs.append((output, control))
    
    # Convert to numpy for easier manipulation
    outputs = np.array(outputs)
    
    # Verify steady-state error is small (relaxed tolerance for discretization)
    assert abs(outputs[-1, 0] - target) < 0.05
    
    # Verify reasonable overshoot (relaxed to 30% for this simple plant model)
    assert np.max(outputs[:, 0]) < target * 1.3


def test_pi_controller_visual():
    """Visual test of PI controller step response."""
    # Create a PI controller
    controller = PIController(
        kp=2.0,
        ki=5.0,
        output_min=-10.0,
        output_max=10.0
    )
    state = controller.init_state()
    
    # Simulation parameters
    dt = 0.01
    sim_time = 2.0
    steps = int(sim_time / dt)
    
    # Initialize arrays
    time = np.linspace(0, sim_time, steps)
    target = np.ones_like(time)
    output = np.zeros_like(time)
    control = np.zeros_like(time)
    
    # Initial state
    current_output = 0.0
    current_state = state
    
    # Run simulation
    for i, t in enumerate(time):
        # Step input at t=0.2s
        current_target = 1.0 if t >= 0.2 else 0.0
        target[i] = current_target
        
        # Compute control
        error = current_target - current_output
        current_control, current_state = controller.step(current_state, error, dt)
        
        # Simple first-order plant model
        current_output += (current_control - current_output) * dt * 5.0
        
        # Store results
        output[i] = current_output
        control[i] = current_control
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot output and target
    ax1.plot(time, target, 'r--', label='Target')
    ax1.plot(time, output, 'b-', label='Output')
    ax1.set_ylabel('Output')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('PI Controller Step Response')
    
    # Plot control signal
    ax2.plot(time, control, 'g-')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Signal')
    ax2.grid(True)
    
    # Save the figure
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'pi_controller_step_response.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    print(f"Visual test complete. Plot saved to {output_path}")


if __name__ == "__main__":
    # Run all test functions
    test_functions = [
        test_pi_controller_step,
        test_pi_controller_anti_windup,
        test_pi_controller_tracking,
        test_pi_controller_visual,
    ]
    
    for test_func in test_functions:
        print(f"Running {test_func.__name__}...")
        test_func()
    
    print("All tests completed.")
