"""Forward run of a traction-motor controller against a warm-started tuning.

This does not optimize; it replays a known parameter set through a chosen
load scenario and plots the domain-randomized ensemble response. Useful for
sanity-checking a tuning before or after running the optimizer.

Three load scenarios are provided as functions; pick one in `main`.
"""

import jax
import jax.numpy as jnp
import scipy.ndimage

from actuator.actuator import Actuator
from actuator.scheduling import Scheduler
from actuator.simulation import Simulation, create_targets, mod_velocity
from actuator.motor import Motor
from actuator.controller import FOCController
from actuator.encoder import Encoder
from actuator.examples.optimize_plot import plot_results


# Each scenario maps a time vector to (targets, weights, load_torque, inertia).
# inertia is returned per-scenario because a position step benefits from a
# heavier simulated load than a braking pulse.

def scenario_abs_brake(times):
    """ABS-braking: a velocity command pulse plus a small position step to test
    balancing under a large opposing load torque."""
    inertia = 0.2
    target_velocity = 60.0  # rad/s
    velocity = jnp.where(jnp.logical_and(times > 0.1, times < 0.3), target_velocity, 0.0)
    velocity = scipy.ndimage.gaussian_filter(velocity, 300)
    # superimpose a position step to probe balancing performance
    step = jnp.where(jnp.logical_and(times > 0.45, times < 0.5), 20, 0.0)
    step = scipy.ndimage.gaussian_filter(step, 150)
    velocity = velocity + step

    load_torque = -jnp.where(jnp.logical_and(times > 0.05, times < 0.2), 130, 40.0)
    load_torque = jnp.where(times > 0.4, 10, load_torque)
    load_torque = scipy.ndimage.gaussian_filter(load_torque, 5)

    targets, weights = create_targets(velocity=velocity)
    return targets, weights, load_torque, inertia


def scenario_step_position(times):
    """Step position control: a ~0.5 m equivalent step with no external load."""
    inertia = 0.4
    target_position = 2.0  # rad
    position = jnp.where(jnp.mod(times / 0.5 - 0.25, 1) > 0.5, 0, target_position)
    position = scipy.ndimage.gaussian_filter(position, 200)

    targets, weights = create_targets(position=position)
    load_torque = jnp.zeros_like(times)
    return targets, weights, load_torque, inertia


def scenario_torque_reversal(times):
    """Max torque reversal at speed: a sudden external torque pulling the wheel
    forward while it tries to hold velocity (an approximate regen probe)."""
    inertia = 0.2
    target_velocity = 50.0  # rad/s
    velocity = jnp.where(jnp.logical_and(times > 0.1, times < 0.5), target_velocity, 0.0)
    velocity = scipy.ndimage.gaussian_filter(velocity, 300)

    load_torque = -jnp.where(jnp.logical_and(times > 0.05, times < 0.3), -20, 130.0)
    load_torque = jnp.where(times > 0.4, 10, load_torque)
    load_torque = scipy.ndimage.gaussian_filter(load_torque, 5)

    targets, weights = create_targets(velocity=velocity)
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
    max_current = 130  # A
    max_voltage = 48.0 * 2 / 3 ** 0.5  # V, peak phase voltage available in the dq frame

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
        scheduler=Scheduler(controller),
    )

    # Domain randomization: scale each named parameter by a uniform factor.
    def uniform(lo, hi):
        return lambda rng: lambda v: v * jax.random.uniform(rng, minval=lo, maxval=hi)

    domain_rand = {
        '__actuator__motor__resistance': uniform(0.8, 1.2),
        '__actuator__motor__inductance_d': uniform(0.8, 1.2),
        '__actuator__motor__inductance_q': uniform(0.8, 1.2),
        '__max_voltage': uniform(0.9, 1.0),
        '__actuator__motor__flux_linkage': uniform(0.95, 1.05),
        '__actuator__motor__inertia': uniform(0.9, 1.0),
        '__actuator__motor__friction': uniform(0.1, 10),
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


# A robust non-scheduling tuning (high velocity-loop ki), warm-started here.
TUNED_PARAMS = {
    '__velocity_ctrl__kp': 8.64779447130406e-6,
    '__velocity_ctrl__ki': 8752.737396012235,
    '__velocity_ctrl__decay': 0.00977337700870426,
    '__velocity_ctrl__kd': 4.116809319821976e-05,
    '__iq_ctrl__kp': 0.9917040926538143,
    '__iq_ctrl__ki': 0.012523761945589589,
    '__id_ctrl__kp': 1.0328488461105612,
    '__id_ctrl__ki': 0.0006387985302455358,
    '__observer__tau_pos': 26.277948121697342,
    '__observer__tau_vel': 129.29932606568744,
    '__voltage_feedforward': 0.99,
    '__current_estimator__tau': 3.237188398612744,
    '__current_estimator__feedforward': 0.031762784135594975,
}


def main() -> None:
    """Replay the tuned parameters through a scenario and plot the result."""
    simulation = create_base_simulation(scenario=scenario_abs_brake)
    simulation = mod_velocity(simulation, 5.0)

    plot_results(
        simulation=simulation,
        initial_params=TUNED_PARAMS,
        optimized_params=TUNED_PARAMS,
    )


if __name__ == "__main__":
    main()