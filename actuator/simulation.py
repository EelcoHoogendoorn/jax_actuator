"""Simulation module.

Wraps a simulatable object (such as an Actuator) together with a scenario
(time vector, position/velocity/torque targets, and load torque), and provides
the rollout and domain-randomization machinery used by the optimizer.
"""
from dataclasses import dataclass, replace
from typing import Any, Dict

import jax
import jax.numpy as jnp

from actuator.actuator import Actuator, ActuatorState
from actuator.jax_utils import register_dataclass, replace_pytree_values


def create_targets(position=None, velocity=None, torque=None):
    """Pack optional position/velocity/torque targets into (targets, weights)
    arrays of shape (T, 3), where weights mark which channels are active.
    """
    a = position if position is not None else velocity if velocity is not None else torque
    targets = jnp.zeros((len(a), 3))
    weights = jnp.zeros((len(a), 3))
    if position is not None:
        targets = targets.at[:, 2].set(position)
        weights = weights.at[:, 2].set(1.0)
    if velocity is not None:
        targets = targets.at[:, 1].set(velocity)
        weights = weights.at[:, 1].set(1.0)
    if torque is not None:
        targets = targets.at[:, 0].set(torque)
        weights = weights.at[:, 0].set(1.0)
    return targets, weights


def mod_velocity(simulation, delta):
    """utility function to modify the velocity of the simulation; see how that impacts the optimum"""
    speed_boost = {
        'state__motor__velocity': delta,
        'state__controller__observer__velocity': delta,
        'targets': simulation.targets.at[:,1].add(delta)
    }
    return replace_pytree_values(simulation, speed_boost)


@register_dataclass
@dataclass
class Simulation:
    """A simulatable object plus a scenario to run it against."""
    actuator: Actuator
    state: ActuatorState
    times: jnp.ndarray
    targets: jnp.ndarray
    weights: jnp.ndarray
    load_torque: jnp.ndarray
    domain: Dict[str, Any]
    key: jax.random.PRNGKey

    def run(self) -> 'SimulationResult':
        """Unroll the simulation steps"""
        # Function to run one step
        @jax.jit
        def step(state, inputs):
            newstate, outputs = self.actuator.step(state, *inputs)
            return newstate, (newstate, outputs)

        # Run simulation
        _, (states, outputs) = jax.lax.scan(
            step,
            # seed every rng_key in the state tree (motor, controller, ...) from self.key
            init=replace_pytree_values(self.state, {'__rng_key': self.key}),
            xs=(self.targets, self.weights, self.load_torque),
        )

        # Unpack the outputs from jax.lax.scan
        v_q, v_d = outputs

        return SimulationResult(
            v_q=v_q,
            v_d=v_d,
            states=states,
        )

    def randomize_domain(self, key) -> "Simulation":
        def sample_dict(rng, params):
            rngs = jax.random.split(rng, len(params))
            return {k: v(rng) for (k, v), rng in zip(params.items(), rngs)}

        domain = sample_dict(key, self.domain)
        # NOTE: need to zero out domain dict first to avoid funny recursions
        return replace_pytree_values(replace(self, domain={}, key=key), domain)

    def eval(self, results):
        raise NotImplementedError


@register_dataclass
@dataclass
class SimulationResult():
    v_q: jnp.ndarray
    v_d: jnp.ndarray
    states: jnp.ndarray
