"""Shared plotting helpers for the optimization examples.

`plot_results` runs a domain-randomized ensemble of a simulation before and
after optimization and overlays the responses, so the effect of tuning is
visible across the randomized population rather than for a single rollout.
"""

from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from actuator.optimize import replace_simulation


def plot_single_simulation(
        simulation,
        result,
        label: str,
        color: Optional[str] = None,
        linestyle: str = '-',
        axs: Optional[List[plt.Axes]] = None,
        plot_targets: bool = True,
        alpha: float = 1,
) -> List[plt.Axes]:
    """Plot a single simulation rollout across modulation, current, torque,
    velocity and position axes.

    Args:
        simulation: Simulation instance the result was produced from.
        result: Simulation result to plot.
        label: Label for the plot legend.
        color: Color for the plot lines.
        linestyle: Line style for the plot.
        axs: Axes to plot on. If None, a new figure and axes are created.
        plot_targets: Whether to overlay the target references.
        alpha: Line transparency (used to fade overlapping ensemble members).

    Returns:
        The list of axes used for plotting.
    """
    t = simulation.times

    if axs is None:
        fig, axs = plt.subplots(5, 1, figsize=(12, 10))

    # Modulation depth (commanded voltage magnitude relative to the limit)
    voltage_magnitude = lambda r: jnp.sqrt(r.v_q ** 2 + r.v_d ** 2) / simulation.actuator.controller.max_voltage
    axs[0].plot(t, voltage_magnitude(result), label=label, color=color, linestyle=linestyle, alpha=alpha)
    axs[0].set_ylabel('Voltage (V)')
    axs[0].set_title('Modulation depth')
    axs[0].grid(True)

    # Current magnitude
    current_magnitude = lambda r: jnp.sqrt(r.states.motor.current_q ** 2 + r.states.motor.current_d ** 2)
    axs[1].plot(t, current_magnitude(result), label=label, color=color, linestyle=linestyle, alpha=alpha)
    axs[1].set_ylabel('Current (A)')
    axs[1].set_title('Current Magnitude')
    axs[1].grid(True)

    # Torque response
    torque = lambda r: jax.vmap(simulation.actuator.motor.torque)(r.states.motor)
    axs[2].plot(t, torque(result), label=label, color=color, linestyle=linestyle, alpha=alpha)
    if plot_targets:
        axs[2].plot(t, simulation.targets[:, 0], 'k--', label='External Torque' if label == 'Reference' else None,
                    alpha=0.5)
    axs[2].set_ylabel('Torque (Nm)')
    axs[2].set_title('Torque Response')
    axs[2].grid(True)

    # Velocity response
    vel = lambda r: r.states.motor.velocity
    axs[3].plot(t, vel(result), label=label, color=color, linestyle=linestyle, alpha=alpha)
    if plot_targets:
        axs[3].plot(t, simulation.targets[:, 1], 'k--', label='Target' if label == 'Reference' else None, alpha=0.5)
    axs[3].set_ylabel('Velocity (rad/s)')
    axs[3].set_title('Velocity Response')
    axs[3].grid(True)

    # Position response (wrapped to [-pi, pi])
    position = lambda r: r.states.motor.position
    wrap = lambda r: jnp.mod(r + jnp.pi, 2 * jnp.pi) - jnp.pi
    axs[4].plot(t, wrap(position(result)), label=label, color=color, linestyle=linestyle, alpha=alpha)
    if plot_targets:
        axs[4].plot(t, wrap(simulation.targets[:, 2]), 'k--', label='Target' if label == 'Reference' else None,
                    alpha=0.5)
    axs[4].set_xlabel('Time (s)')
    axs[4].set_ylabel('Position (rad)')
    axs[4].set_title('Position Response')
    axs[4].grid(True)

    # De-duplicate legend entries per axis
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    return axs


def unpack_vmapped_pytree(vmapped_pytree, batch_size):
    """Unpack a vmapped PyTree into a list of PyTrees, one per batch entry.

    Args:
        vmapped_pytree: A PyTree where each leaf has a leading batch dimension.
        batch_size: The size of the batch dimension.

    Returns:
        A list of PyTrees, each corresponding to one entry in the batch.
    """
    return [jax.tree_util.tree_map(lambda x: x[i], vmapped_pytree) for i in range(batch_size)]


def plot_results(
        simulation,
        initial_params: Dict[str, float],
        optimized_params: Dict[str, float],
        save_path: Optional[str] = None,
        show: bool = True,
) -> None:
    """Plot a domain-randomized ensemble before and after optimization.

    Args:
        simulation: Base simulation instance.
        initial_params: Parameter values before optimization.
        optimized_params: Parameter values after optimization.
        save_path: If given, the figure is written here.
        show: Whether to display the figure interactively.
    """
    sim_before = replace_simulation(simulation, initial_params)
    sim_after = replace_simulation(simulation, optimized_params)

    fig, axs = plt.subplots(5, 1, figsize=(12, 10))

    # Run an ensemble of domain-randomized rollouts for each parameter set
    seed = 43
    N = 30
    keys = jax.random.split(jax.random.PRNGKey(seed), N)

    def ensemble_rollout(sim):
        def run_one(key):
            s = sim.randomize_domain(key)
            return s, s.run()
        return run_one

    sims, results = jax.jit(jax.vmap(ensemble_rollout(sim_before)))(keys)
    for r, s in unpack_vmapped_pytree((results, sims), N):
        plot_single_simulation(s, r, 'Before Optimization',
                               alpha=2 / N, color='C0', linestyle='--', axs=axs, plot_targets=False)
    sims, results = jax.jit(jax.vmap(ensemble_rollout(sim_after)))(keys)
    for r, s in unpack_vmapped_pytree((results, sims), N):
        plot_single_simulation(s, r, 'After Optimization',
                               alpha=2 / N, color='C1', axs=axs, plot_targets=True)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()