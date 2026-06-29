"""Tests for the motor module."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any

import jax
import numpy as np
import matplotlib.pyplot as plt
from actuator.motor import MotorState, Motor

# Test parameters
SIMULATION_TIME = 0.5  # seconds
DT = 1e-4  # seconds (10 kHz)
SAMPLE_INTERVAL = 1  # Only plot every N points for better visibility
t = np.arange(0, SIMULATION_TIME, DT)



def test_mosfet():
    import jax.numpy as jnp
    P = lambda t: 10.00 * jnp.pow(t, -0.5663)
    # R*A^2 =
    R = 4.5e-3
    A = lambda t: jnp.sqrt(10.00/R * jnp.pow(t, -0.5663))
    t = jnp.logspace(-3, 2)
    import matplotlib.pyplot as plt
    plt.loglog(t, A(t))
    plt.show()


def test_harmonic():
    def dedent(aa):
        poles = 40
        slots = 32
        fp = [1 / 2, 1, 2]
        fs = [1, 2, 3]
        h = lambda f, a: np.sin(aa * f) * a
        return sum(h(poles*q,1) for q in fp)+sum(h(slots*q,1) for q in fs)
    aa = np.linspace(0, 2*np.pi, 8000)
    plt.plot(dedent(aa))
    plt.show()


def run_motor_simulation(
    voltage_func: callable,
    motor: Motor = None
):
    """Run a motor simulation with the given voltage profile.
    
    Args:
        voltage_func: Function that takes time array and returns (voltage_d, voltage_q)
        sim_time: Total simulation time in seconds
        dt: Time step in seconds
        motor: Motor instance (defaults to Motor.create_default() if None)
        
    Returns:
        SimulationResult containing all simulation data
    """
    if motor is None:
        motor = Motor.create_default()

    state = MotorState()

    # Generate voltage profiles
    voltage_d, voltage_q = voltage_func(t)

    @jax.jit
    def update(state, inputs):
        state = motor.update(state, *inputs, dt=DT)
        return state, state
    # Run simulation
    _, states =jax.lax.scan(update, xs=(voltage_d, voltage_q, voltage_q*0), init=state)
    return states


def step_voltage(t: np.ndarray, step_time: float = 0.1, amplitude: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate step voltage profile."""
    voltage_q = np.zeros_like(t)
    step_start = int(step_time / DT)
    voltage_q[step_start:] = amplitude
    return np.zeros_like(t), voltage_q


def step_voltage(t: np.ndarray, step_time: float = 0.1, amplitude: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate HFI profile."""
    a = 2 * np.pi * t*600
    c, s = np.cos(a), np.sin(a)
    return c, s


def plot_simulation_results(result, title: str = "Motor Response"):
    """Plot simulation results."""

    # Downsample for better plotting performance
    sample_indices = slice(None, None, SAMPLE_INTERVAL)
    t_plot = t[sample_indices]
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot position and velocity
    ax1.plot(t_plot, result.position[sample_indices], label='Position (rad)')
    ax1.set_ylabel('Position (rad)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t_plot, result.velocity[sample_indices], label='Velocity (rad/s)', color='orange')
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.legend()
    ax2.grid(True)

    # Plot currents
    ax3.plot(t_plot, result.current_q[sample_indices], label='Current Q (A)')
    ax3.plot(t_plot, result.current_d[sample_indices], label='Current D (A)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Current (A)')
    ax3.legend()
    ax3.grid(True)

    plt.suptitle(title)
    plt.tight_layout()

    return fig

def plot_circle(result, title):

    # Downsample for better plotting performance
    sample_indices = slice(None, None, SAMPLE_INTERVAL)
    t_plot = t[sample_indices]

    vd, vq = step_voltage(t)
    id, iq = result.current_d, result.current_q

    i = id + iq * 1j
    v = vd + vq * 1j
    p = i/v
    a = np.mean(np.angle(p[100:], deg=True))
    print('angle', a)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(vq[sample_indices], vd[sample_indices], label='Current D (A)')
    ax1.set_xlabel('q (V)')
    ax1.set_ylabel('d (V)')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    # Plot currents
    ax2.plot(iq[sample_indices], id[sample_indices], label='Current D (A)')
    ax2.set_xlabel('q (A)')
    ax2.set_ylabel('d (A)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    plt.suptitle(title)
    # plt.tight_layout()

    return fig


def test_step_response_visual():
    """Visual test of motor response to a step input in voltage_q."""
    # Run simulation with step voltage
    result = run_motor_simulation(
        voltage_func=lambda t: step_voltage(t, step_time=0.1, amplitude=10.0),
    )
    
    # Plot results
    # fig = plot_simulation_results(result, 'Motor Step Response (10V q-axis step at t=0.1s)')
    fig = plot_circle(result, 'motor hfi reponse')
    # Save the figure
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'motor_step_response.png'
    fig.savefig(output_path)
    plt.show()
    plt.close(fig)  # Close the figure to free memory
    print(f"Visual test complete. Plot saved to {output_path}")


# def test_step_response_text():
#     """Text-based test of motor response to step voltage_q input."""
#     # Run simulation with sinusoidal voltage
#     result = run_motor_simulation(
#         voltage_func=lambda t: step_voltage(t, step_time=0.1, amplitude=10.0),
#     )
#
#     # Print results in CSV format
#     print("time,position,velocity,current_q")
#     for i in range(0, len(t), SAMPLE_INTERVAL * 10):
#         print(f"{t[i]:.6f},{result.position[i]:.6f},"
#               f"{result.velocity[i]:.6f},{result.current_qs[i]:.6f}")

if __name__ == "__main__":
    test_step_response_visual()
    # test_step_response_text()
