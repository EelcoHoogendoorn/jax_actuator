"""Tests for the controller module with encoder integration."""

from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Import the controller and motor modules
from actuator.motor import MotorState, Motor
from actuator.encoder import Encoder
from actuator.observer import EMAPositionVelocityObserver
from actuator.controller import FOCController
from actuator.actuator import Actuator
from actuator.scheduling import Scheduler
from actuator.simulation import SimulationResult, Simulation, create_targets

# Set random seed for reproducibility
jax.config.update('jax_enable_x64', True)
SAMPLE_INTERVAL = 10


def run_closed_loop_simulation(
    target_velocity_func: Callable[[float], float],
    sim_time: float,
    dt: float,
) -> SimulationResult:
    """Run a closed-loop motor simulation with the given target velocity profile.

    Args:
        target_velocity_func: Function that takes time and returns target velocity (rad/s)
        sim_time: Total simulation time in seconds
        dt: Time step in seconds

    Returns:
        SimulationResult containing all simulation data
    """
    rng_key = jax.random.PRNGKey(42)

    motor = Motor.create_default()
    encoder = Encoder(
        resolution=2**13,
        noise_std=0.5,
        phase_offset=0.0
    )

    controller = FOCController.create_default(
        motor=motor,
        encoder=encoder,
    )

    actuator = Actuator(
        motor=motor,
        controller=controller,
        scheduler=Scheduler(c0=controller),
        dt=dt,
    )

    # Time array and pre-compute target velocities
    t = jnp.arange(0, sim_time, dt)
    target_velocities = jnp.array([target_velocity_func(ti) for ti in t])
    targets, weights = create_targets(velocity=target_velocities)

    sim = Simulation(
        actuator=actuator,
        state=actuator.init_state(rng_key),
        times=t,
        targets=targets,
        weights=weights,
        load_torque=jnp.zeros_like(t),
        domain={},
        key=rng_key,
    )
    return sim.run()

def plot_closed_loop_results(result: SimulationResult, title: str = "Closed-Loop Response"):
    """Plot closed-loop simulation results."""
    t = result.time
    
    # Downsample for better plotting performance
    sample_indices = slice(None, None, SAMPLE_INTERVAL)
    t_plot = t[sample_indices]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Plot velocity
    ax1.plot(t_plot, result.target_velocities[sample_indices], 'r--', label='Target')
    ax1.plot(t_plot, result.velocities[sample_indices], 'b-', label='Actual')
    ax1.set_ylabel('Velocity (rad/s)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Velocity Response')
    
    # Plot position
    ax2.plot(t_plot, result.positions[sample_indices])
    ax2.set_ylabel('Position (rad)')
    ax2.grid(True)
    ax2.set_title('Position')
    
    # Plot currents
    ax3.plot(t_plot, result.current_qs[sample_indices], label='Iq (A)')
    ax3.plot(t_plot, result.current_ds[sample_indices], label='Id (A)')
    ax3.set_ylabel('Current (A)')
    ax3.legend()
    ax3.grid(True)
    ax3.set_title('Currents')
    
    # Plot voltages
    ax4.plot(t_plot, result.voltage_qs[sample_indices], label='Vq (V)')
    ax4.plot(t_plot, result.voltage_ds[sample_indices], label='Vd (V)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Voltage (V)')
    ax4.legend()
    ax4.grid(True)
    ax4.set_title('Control Voltages')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def test_step_response_visual(rng_key: Optional[jax.random.PRNGKey] = None):
    """Visual test of controller response to a step in target velocity with encoder.
    
    Args:
        rng_key: Optional random number generator key.
    """
    # Define target velocity profile (step at t=0.2s)
    def target_velocity(t):
        return 10.0 if t >= 0.2 else 0.0  # 10 rad/s step

    # Run simulation
    result = run_closed_loop_simulation(
        target_velocity_func=target_velocity,
        sim_time=1.0,
        dt=1e-4,
    )
    
    # Extract data from result
    n = len(result.v_q)
    t = jnp.arange(n) * 1e-4
    velocities = result.states.motor.velocity
    positions = result.states.motor.position
    current_qs = result.states.motor.current_q

    # Create plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    ax1.plot(t[::SAMPLE_INTERVAL],
             velocities[::SAMPLE_INTERVAL],
             'b-', label='Velocity')
    ax1.set_ylabel('Velocity (rad/s)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Velocity Response')

    ax2.plot(t[::SAMPLE_INTERVAL],
             positions[::SAMPLE_INTERVAL],
             'b-', label='Position')
    ax2.set_ylabel('Position (rad)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Position')

    ax3.plot(t[::SAMPLE_INTERVAL],
             current_qs[::SAMPLE_INTERVAL],
             'b-', label='Q-axis Current')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Current (A)')
    ax3.legend()
    ax3.grid(True)
    ax3.set_title('Q-axis Current')

    V = jnp.sqrt(result.v_q**2 + result.v_d**2)[::SAMPLE_INTERVAL]
    ax4.plot(t[::SAMPLE_INTERVAL], V, label='V (V)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Voltage (V)')
    ax4.legend()
    ax4.grid(True)
    ax4.set_title('Control Voltages')

    plt.suptitle('Step Response')
    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'step_response.png'
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Visual test complete. Plot saved to {output_path}")

def test_visual_multi_step_response(rng_key: Optional[jax.random.PRNGKey] = None):
    """Visual test of controller response to multiple step changes in target velocity.
    
    This test shows how the controller with encoder and velocity filter
    handles direction changes and varying speed commands.
    
    Args:
        rng_key: Optional random number generator key.
    """
    def target_velocity(t):
        if t < 0.5:
            return 0.0      # Start at zero
        elif t < 1.5:
            return 20.0     # Step up to 20 rad/s
        elif t < 2.5:
            return -10.0    # Step down to -10 rad/s
        elif t < 3.5:
            return 5.0      # Step up to 5 rad/s
        else:
            return 0.0      # Back to zero

    result = run_closed_loop_simulation(
        target_velocity_func=target_velocity,
        sim_time=4.0,
        dt=1e-4,
    )

    # Extract data from result
    n = len(result.v_q)
    t = jnp.arange(n) * 1e-4
    velocities = result.states.motor.velocity
    positions = result.states.motor.position

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(t[::SAMPLE_INTERVAL],
             velocities[::SAMPLE_INTERVAL],
             'b-', label='Velocity')
    ax1.set_ylabel('Velocity (rad/s)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Multi-Step Velocity Response')

    ax2.plot(t[::SAMPLE_INTERVAL],
             positions[::SAMPLE_INTERVAL],
             'b-', label='Position')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (rad)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Position')

    plt.suptitle('Multi-Step Response')
    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'multi_step_response.png'
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Visual test complete. Plot saved to {output_path}")

if __name__ == "__main__":
    # Create a single RNG key at the start and split it for each test
    main_key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(main_key)
    
    print("Running single step response test...")
    test_step_response_visual(key1)
    
    print("\nRunning multi-step response test...")
    test_visual_multi_step_response(key2)
