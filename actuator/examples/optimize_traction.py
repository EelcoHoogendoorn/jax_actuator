"""Optimize a traction controller for an ABS-braking scenario: track a velocity
command pulse (with a superimposed position step) while rejecting a large
opposing load-torque impulse.

Several qualitatively different optima coexist for this scenario, and persist
under domain randomization: a high voltage-feedforward solution with most
gains near zero, lower-feedforward solutions with more active control, and a
notably robust family with large velocity-loop ki and current-estimator tau.
This example warm-starts from the robust high-ki family. An alternate
step-position scenario is also provided.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import scipy.ndimage

from actuator.actuator import Actuator
from actuator.scheduling import Scheduler
from actuator.simulation import Simulation, create_targets, mod_velocity
from actuator.motor import Motor
from actuator.controller import FOCController
from actuator.encoder import Encoder
from actuator.utils import angle_difference_wrap
from actuator.optimize import replace_simulation, optimize_simulation
from actuator.examples.optimize_plot import plot_results


MAX_CURRENT = 130  # A; shared by the controller limit and the cost over-current threshold


# Each scenario maps a time vector to (targets, weights, load_torque, inertia).

def scenario_abs_brake(times):
    """Velocity command pulse plus a superimposed position step, under a large
    opposing load-torque impulse."""
    inertia = 0.2
    target_velocity = 50.0  # rad/s
    velocity = jnp.where(jnp.logical_and(times > 0.1, times < 0.3), target_velocity, 0.0)
    velocity = scipy.ndimage.gaussian_filter(velocity, 250)
    step = jnp.where(jnp.logical_and(times > 0.45, times < 0.5), 30, 0.0)
    step = scipy.ndimage.gaussian_filter(step, 150)
    velocity = velocity + step

    load_torque = -jnp.where(jnp.logical_and(times > 0.05, times < 0.2), 130, -10.0)
    load_torque = scipy.ndimage.gaussian_filter(load_torque, 5)

    targets, weights = create_targets(velocity=velocity)
    return targets, weights, load_torque, inertia


def scenario_step_position(times):
    """A ~0.5 m equivalent position step with no external load."""
    inertia = 0.4
    target_position = 2.0  # rad
    position = jnp.where(jnp.mod(times / 0.5 - 0.25, 1) > 0.5, 0, target_position)
    position = scipy.ndimage.gaussian_filter(position, 200)

    targets, weights = create_targets(position=position)
    load_torque = jnp.zeros_like(times)
    return targets, weights, load_torque, inertia


def create_base_simulation(scenario=scenario_abs_brake) -> Simulation:
    """Build a traction-motor simulation for the given load scenario.

    Args:
        scenario: A scenario function mapping times -> (targets, weights,
            load_torque, inertia).

    Returns:
        A configured Simulation instance.
    """
    dt = 1 / 8000
    simulation_time = 0.8  # seconds
    times = jnp.arange(0.0, simulation_time, dt)

    targets, weights, load_torque, inertia = scenario(times)

    Kt = 1.219
    pole_pairs = 16
    motor = Motor(
        resistance=0.24,
        inductance_d=0.0003,
        inductance_q=0.00033,
        Kt=Kt,
        pole_pairs=pole_pairs,
        slots=36,
        inertia=inertia,
        friction=8e-3,
        hysteresis=0.17,
        torque_static=0.3,
    )

    # Current/voltage limits are properties of the power electronics, not the motor.
    max_current = MAX_CURRENT
    max_voltage = 58.0 * 2 / jnp.sqrt(3)  # V, peak phase voltage available in the dq frame

    encoder = Encoder(resolution=2 ** 14, noise_std=2 ** (14 - 12.5) / jnp.sqrt(12))  # ~IC-MU class

    controller = FOCController.create_default(
        motor=motor,
        encoder=encoder,
        max_voltage=max_voltage,
        max_current=max_current,
        max_velocity=100,
    )

    actuator = Actuator(
        motor=motor,
        controller=controller,
        dt=dt,
        scheduler=Scheduler(c0=controller),
    )

    # Domain randomization: scale each named parameter by a uniform factor.
    def uniform(lo, hi):
        return lambda rng: lambda v: v * jax.random.uniform(rng, minval=lo, maxval=hi)

    domain_rand = {
        '__actuator__motor__resistance': uniform(0.8, 1.2),
        '__actuator__motor__inductance_d': uniform(0.8, 1.2),
        '__actuator__motor__inductance_q': uniform(0.8, 1.2),
        '__actuator__controller__max_voltage': uniform(0.9, 1.0),
        '__actuator__motor__flux_linkage': uniform(0.95, 1.05),
        '__actuator__motor__inertia': uniform(0.9, 1.0),
        '__actuator__motor__friction': uniform(0.5, 2),
        '__actuator__motor__dedent_offset': uniform(0.0, 1.0),
    }

    return Simulation(
        actuator=actuator,
        state=actuator.init_state(jax.random.PRNGKey(0)),
        times=times,
        targets=targets,
        weights=weights,
        load_torque=load_torque,
        domain=domain_rand,
        key=jax.random.PRNGKey(0),
    )


def cost(simulation, result):
    """Cost trading off velocity tracking, ohmic loss, torque smoothness and
    over-current penalty."""
    targets = simulation.targets
    weights = simulation.weights
    t_target, v_target, p_target = targets.T
    t_weight, v_weight, p_weight = weights.T
    motor = result.states.motor

    # Tracking error (velocity-weighted; position term disabled for this scenario)
    errors = (motor.velocity - v_target) * v_weight
    iae = jnp.mean(jnp.abs(errors))

    # Over-current penalty above the limit
    i2 = motor.current_q ** 2 + motor.current_d ** 2
    overcurrent = jnp.maximum(jnp.sqrt(i2) - MAX_CURRENT, 0)
    overcurrent_cost = jnp.mean(overcurrent ** 2)

    # Resistive loss (keep currents no larger than necessary)
    ohmic = jnp.mean(i2)

    # Torque jerk, weighted toward the settled phase: jittery torque is audible
    settled = jnp.where(simulation.times > 0.6, 1, 1e-2)
    torque = jax.vmap(simulation.actuator.motor.torque)(result.states.motor)
    noise1 = jnp.mean(jnp.abs(jnp.diff(torque * settled, n=1)))
    noise2 = jnp.mean(jnp.abs(jnp.diff(torque * settled, n=2)))

    return 1e2 * iae + 1e-4 * ohmic + 5e2 * noise1 + 5e2 * noise2 + 1e2 * overcurrent_cost


# Robust warm-start from the high velocity-loop-ki family. A trailing
# underscore marks a parameter that is applied to the simulation but held
# fixed (not optimized).
INITIAL_PARAMS = {
    '__velocity_ctrl__kp': 0.001,
    '__velocity_ctrl__ki': 20000,
    '__velocity_ctrl__decay': 0.01,
    '__velocity_ctrl__kd': 1e-5,
    '__iq_ctrl__kp': 1.5,
    '__iq_ctrl__ki': 1e-2,
    '__id_ctrl__kp': 8.0,
    '__id_ctrl__ki': 1e-3,
    '__observer__tau_pos': 12.,
    '__observer__tau_vel': 80.,
    '__voltage_feedforward_': 0.95,  # held fixed
    '__current_estimator__tau': 28.,
    '__current_estimator__feedforward': 0.01,
}


def main() -> None:
    """Run the optimization and plot the before/after ensemble response."""
    simulation = create_base_simulation()

    # Apply all warm-start values; optimize only those without a trailing underscore.
    simulation = replace_simulation(simulation, {k.rstrip('_'): v for k, v in INITIAL_PARAMS.items()})
    opt_params = {k: v for k, v in INITIAL_PARAMS.items() if not k.endswith('_')}
    fixed_params = {k.rstrip('_'): v for k, v in INITIAL_PARAMS.items() if k.endswith('_')}

    simulation = mod_velocity(simulation, -5.0)

    print("Starting optimization...")
    print("-" * 50)

    best_params, best_fitness = optimize_simulation(
        simulation=simulation,
        cost=cost,
        initial_params=opt_params,
        pop_size=64,
        num_generations=100,
        seed=0,
    )
    best_params.update(fixed_params)

    print("\nOptimization complete!")
    print("Optimized parameters:")
    for k, v in best_params.items():
        print(f"{k}: {float(v):.6f}")
    print('Copy-paste form:')
    print({k: float(v) for k, v in best_params.items()})
    print(f"Best fitness: {best_fitness:.6f}")

    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    plot_results(
        simulation=simulation,
        initial_params={k.rstrip('_'): v for k, v in INITIAL_PARAMS.items()},
        optimized_params=best_params,
        save_path=output_dir / 'optimize_traction.png',
    )

    return best_params


if __name__ == "__main__":
    main()