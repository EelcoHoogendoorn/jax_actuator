"""Optimize a controller for a position-step move while rejecting an external
load-torque impulse.

Several qualitatively different optima coexist (a high-feedforward /
near-zero-gain family and lower-feedforward families with more active control).
This example warm-starts from a position/velocity cascade and optimizes position
tracking while respecting current limits and torque smoothness. An alternate
scenario (`scenario_torque_step`) instead tracks a commanded torque setpoint.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import scipy.ndimage

from actuator.actuator import Actuator
from actuator.scheduling import Scheduler
from actuator.simulation import Simulation, create_targets
from actuator.motor import Motor
from actuator.controller import FOCController
from actuator.encoder import Encoder
from actuator.optimize import replace_simulation, optimize_simulation
from actuator.utils import angle_difference_wrap
from actuator.examples.optimize_plot import plot_results


# Each scenario maps a time vector to (targets, weights, load_torque).

def scenario_torque_step(times):
    """A 10 Nm torque step applied at t = 0.2 s, no external load."""
    target_torque = (times > 0.2) * 10.0
    targets, weights = create_targets(torque=target_torque)
    load_torque = jnp.zeros_like(times)
    return targets, weights, load_torque


def scenario_position_step(times):
    """A 45 degree position step at t = 0.25 s under a brief load-torque impulse."""
    target_position = -jnp.pi / 2 / 2
    position = jnp.where(times > 0.25, target_position, 0.0)
    position = scipy.ndimage.gaussian_filter(position, 400)
    load_torque = -jnp.where(jnp.logical_and(times > 0.05, times < 0.06), 400, 0)
    load_torque = scipy.ndimage.gaussian_filter(load_torque, 5)
    targets, weights = create_targets(position=position)
    return targets, weights, load_torque


def create_base_simulation(scenario=scenario_position_step) -> Simulation:
    """Build a simulation for the given scenario.

    Args:
        scenario: A scenario function mapping times -> (targets, weights, load_torque).

    Returns:
        A configured Simulation instance.
    """
    Kt = 1.2
    pole_pairs = 20
    motor = Motor(
        resistance=0.35,
        inductance_d=0.00027,
        inductance_q=0.0003,
        Kt=Kt,
        slots=36,
        pole_pairs=pole_pairs,
        inertia=0.2,
        friction=1e-4,
        hysteresis=0.21,
        torque_static=0.3,
    )

    # Current/voltage limits are properties of the power electronics, not the motor.
    max_current = 130  # A
    max_voltage = 58.0 * 2 / jnp.sqrt(3)  # V, peak phase voltage available in the dq frame

    encoder = Encoder(resolution=2 ** 14, noise_std=2 ** (14 - 12.5) / jnp.sqrt(12))  # ~IC-MU class

    controller = FOCController.create_default(
        motor=motor,
        encoder=encoder,
        max_voltage=max_voltage,
        max_current=max_current,
        max_velocity=20,
    )

    actuator = Actuator(
        motor=motor,
        controller=controller,
        scheduler=Scheduler(controller),
        dt=1 / 8000,
    )

    simulation_time = 0.6  # seconds
    times = jnp.arange(0.0, simulation_time, actuator.dt)
    targets, weights, load_torque = scenario(times)

    # Domain randomization: scale each named parameter by a uniform factor.
    def uniform(lo, hi):
        return lambda rng: lambda v: v * jax.random.uniform(rng, minval=lo, maxval=hi)

    domain_rand = {
        '__actuator__motor__resistance': uniform(0.9, 1.2),
        '__actuator__motor__inductance_d': uniform(0.8, 1.2),
        '__actuator__motor__inductance_q': uniform(0.8, 1.2),
        '__actuator__controller__max_voltage': uniform(0.9, 1.0),
        '__actuator__motor__flux_linkage': uniform(0.95, 1.05),
        '__actuator__motor__inertia': uniform(0.8, 1.2),
        '__actuator__motor__friction': uniform(0.1, 10),
        '__actuator__motor__dedent_offset': uniform(0.0, 1.0),
    }

    return Simulation(
        state=actuator.init_state(jax.random.PRNGKey(0)),
        actuator=actuator,
        times=times,
        targets=targets,
        weights=weights,
        load_torque=load_torque,
        domain=domain_rand,
        key=jax.random.PRNGKey(0),
    )


def cost(simulation, result):
    """Cost trading off position tracking, ohmic loss, torque smoothness and
    over-current penalty."""
    targets = simulation.targets
    weights = simulation.weights
    t_target, v_target, p_target = targets.T
    t_weight, v_weight, p_weight = weights.T
    motor = result.states.motor

    error_p = angle_difference_wrap(motor.position - p_target) * p_weight
    iae = jnp.mean(error_p ** 2) * 10

    i2 = motor.current_q ** 2 + motor.current_d ** 2
    overcurrent = jnp.maximum(jnp.sqrt(i2) - (simulation.actuator.controller.max_current + 10), 0)
    overcurrent_cost = jnp.mean(overcurrent ** 2)

    # Penalize loss and torque jitter hardest over the final, settled quarter
    settled = len(t_target) // 4 * 3
    ohmic = jnp.mean(i2[settled:])

    torque = motor.current_q * simulation.actuator.motor.Kt
    noise = jnp.mean(jnp.abs(jnp.diff(torque, n=2)))

    return 2e1 * iae + 1e0 * ohmic + 3e-2 * noise + 1e0 * overcurrent_cost


# Warm-start: a position/velocity cascade with current inner loops.
INITIAL_PARAMS = {
    '__position_ctrl__kp': 60.,
    '__position_ctrl__ki': 0.1,
    '__position_ctrl__kd': 0.1,
    '__velocity_ctrl__kp': 30.,
    '__velocity_ctrl__ki': .1,
    '__velocity_ctrl__max_rate': 1e1,
    '__iq_ctrl__kp': 1.0,
    '__iq_ctrl__ki': 1e-3,
    '__iq_ctrl__max_rate': 5e0,
    '__id_ctrl__kp': 1.0,
    '__id_ctrl__ki': 1e-3,
    '__id_ctrl__max_rate': 1e0,
    '__observer__tau_pos': 1.,
    '__observer__tau_vel': 1.,
    '__voltage_feedforward': 0.5,
    '__current_estimator__tau': 0.01,
    '__current_estimator__feedforward': 0.5,
}


def main() -> None:
    """Run the optimization and plot the before/after ensemble response."""
    simulation = create_base_simulation()

    print("Parameters being optimized:")
    print(INITIAL_PARAMS)
    print("Starting optimization...")
    print("-" * 50)

    simulation = replace_simulation(simulation, INITIAL_PARAMS)

    best_params, best_fitness = optimize_simulation(
        simulation=simulation,
        cost=cost,
        initial_params=INITIAL_PARAMS,
        pop_size=64,
        num_generations=50,
        seed=42,
    )

    print("\nOptimization complete!")
    print("Optimized parameters:")
    for k, v in best_params.items():
        print(f"{k}: {float(v):.6f}")
    print(f"Best fitness: {best_fitness:.6f}")

    output_dir = Path(__file__).parent.parent / 'output'
    plot_results(
        simulation=simulation,
        initial_params=INITIAL_PARAMS,
        optimized_params=best_params,
        save_path=output_dir / 'optimize_torque.png',
    )

    return best_params


if __name__ == "__main__":
    main()