"""Optimize a torque-scheduled controller for a position-step move while
rejecting an external load-torque impulse.

This uses a `TorqueScheduler` that blends two controllers (`c0` for fine,
near-zero-error control and `c1` for large-error / high-current transients).
Gain scheduling over torque lets the two regimes be tuned independently.
"""

import jax
import jax.numpy as jnp
import scipy.ndimage

from actuator.actuator import Actuator
from actuator.scheduling import TorqueScheduler
from actuator.simulation import Simulation, create_targets
from actuator.motor import Motor
from actuator.controller import FOCController
from actuator.encoder import Encoder
from actuator.optimize import replace_simulation, optimize_simulation
from actuator.utils import angle_difference_wrap
from actuator.examples.optimize_plot import plot_results


MAX_CURRENT = 130  # A; shared by the controller limit and the cost over-current threshold


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
    """Build a torque-scheduled simulation for the given scenario.

    Args:
        scenario: A scenario function mapping times -> (targets, weights, load_torque).

    Returns:
        A configured Simulation instance.
    """
    Kt = 1.214
    pole_pairs = 20
    motor = Motor(
        resistance=0.35,  # dq-frame values
        inductance_d=0.00028,
        inductance_q=0.0003,
        Kt=Kt,
        pole_pairs=pole_pairs,
        slots=36,
        inertia=0.2,
        friction=12e-3,
        hysteresis=0.21,
        torque_static=0.3,
    )

    # Current/voltage limits are properties of the power electronics, not the motor.
    # max_current is the commanded limit; leave room for overshoot from regen and dynamics.
    max_current = MAX_CURRENT
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
        dt=1 / 8000,
        scheduler=TorqueScheduler(controller, controller),
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
        '__actuator__motor__flux_linkage': uniform(0.95, 1.05),
        '__actuator__motor__inertia': uniform(0.8, 1.2),
        '__actuator__motor__friction': uniform(0.1, 10),
        '__max_voltage': uniform(0.8, 1.0),
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
    iae = jnp.mean(jnp.abs(error_p)) * 100

    i2 = motor.current_q ** 2 + motor.current_d ** 2
    overcurrent = jnp.maximum(jnp.sqrt(i2) - MAX_CURRENT, 0)
    overcurrent_cost = jnp.mean(overcurrent ** 2)

    ohmic = jnp.mean(i2)

    settled = jnp.where(simulation.times > 0.4, 1, 1e-2)
    torque = motor.current_q * simulation.actuator.motor.Kt
    noise1 = jnp.mean(jnp.abs(jnp.diff(torque * settled, n=1)))
    noise2 = jnp.mean(jnp.abs(jnp.diff(torque * settled, n=2)))

    return 2e1 * iae + 1e-4 * ohmic + 3e1 * noise1 + 3e1 * noise2 + 1e1 * overcurrent_cost


# Warm-start for the two torque-scheduled controllers (c0 fine control, c1
# high-current transients). A trailing underscore marks a parameter that is
# applied but held fixed (not optimized) — here many of the inner-loop current
# gains are pinned to get well-behaved current maxing.
INITIAL_PARAMS = {
    '__c0__position_ctrl__kp': 300.,
    '__c0__position_ctrl__ki_': 5.,
    '__c0__position_ctrl__kd': 1,
    '__c0__velocity_ctrl__kp': .4,
    '__c0__velocity_ctrl__ki': .1,
    '__c0__velocity_ctrl__max_rate': 5e2,
    '__c0__iq_ctrl__kp_': 1e-1,
    '__c0__iq_ctrl__ki_': 1e-3,
    '__c0__iq_ctrl__max_rate': 5e-1,
    '__c0__id_ctrl__kp_': 1e-1,
    '__c0__id_ctrl__ki_': 1e-3,
    '__c0__id_ctrl__max_rate': 5e0,
    '__c0__observer__tau_pos_': 0.1,
    '__c0__observer__tau_vel': 10.,
    '__c0__voltage_feedforward': 0.1,
    '__c0__current_estimator__tau': 1.0,
    '__c0__current_estimator__feedforward_': 0.1,

    '__c1__position_ctrl__kp': 300.,
    '__c1__position_ctrl__ki': 2.0,
    '__c1__position_ctrl__kd': 4,
    '__c1__velocity_ctrl__kp': 7.,
    '__c1__velocity_ctrl__ki': 5e-1,
    '__c1__velocity_ctrl__max_rate': 5e4,
    '__c1__iq_ctrl__kp_': 2.0,
    '__c1__iq_ctrl__ki_': 1e-3,
    '__c1__iq_ctrl__max_rate_': 5e2,
    '__c1__id_ctrl__kp_': 2.0,
    '__c1__id_ctrl__ki_': 1e-3,
    '__c1__id_ctrl__max_rate_': 5e2,
    '__c1__observer__tau_pos': 0.001,
    '__c1__observer__tau_vel': 0.001,
    '__c1__voltage_feedforward': 1.1,  # needed to hit peak currents stably
    '__c1__current_estimator__tau': 0.1,
    '__c1__current_estimator__feedforward_': 0.01,

    '__scheduler__deadband_': 0.001,  # held fixed
}


def main() -> None:
    """Run the optimization and plot the before/after ensemble response."""
    simulation = create_base_simulation()

    # Apply all warm-start values; optimize only those without a trailing underscore.
    all_params = {k.rstrip('_'): v for k, v in INITIAL_PARAMS.items()}
    simulation = replace_simulation(simulation, all_params)
    opt_params = {k: v for k, v in INITIAL_PARAMS.items() if not k.endswith('_')}

    print("Parameters being optimized:")
    print(opt_params)
    print("Starting optimization...")
    print("-" * 50)

    best_params, best_fitness = optimize_simulation(
        simulation=simulation,
        cost=cost,
        initial_params=opt_params,
        pop_size=64,
        num_generations=100,
        seed=42,
    )

    print("\nOptimization complete!")
    print("Optimized parameters:")
    for k, v in best_params.items():
        print(f"{k}: {float(v):.6f}")
    print(f"Best fitness: {best_fitness:.6f}")

    plot_results(
        simulation=simulation,
        initial_params=all_params,
        optimized_params=best_params,
    )

    return best_params


if __name__ == "__main__":
    main()