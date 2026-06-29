"""Generic optimization utils"""


import jax
import jax.numpy as jnp

from evosax.algorithms import CMA_ES
from typing import Dict, Any, Tuple

from actuator.simulation import Simulation
from actuator.jax_utils import replace_pytree_values
from actuator.utils import softplus, softplus_norm


def replace_simulation(sim: Simulation, params: Dict[str, jnp.ndarray]) -> Simulation:
    return replace_pytree_values(sim, params)


def optimize_simulation(
		simulation,
		cost: callable,
		initial_params: dict,
		pop_size: int = 20,
		num_generations: int = 50,
		seed: int = 42
) -> Tuple[Dict[str, float], float]:
	"""Optimize controller parameters using CMA-ES.

	Args:
		simulation: The simulation to optimize
		cost: The cost function to optimize
		initial_params: Initial parameters dictionary
		pop_size: Population size for CMA-ES
		num_generations: Number of generations to run
		seed: Random seed

	Returns:
		Optimized controller parameters
		best score
	"""
	# Initialize random key
	key = jax.random.PRNGKey(seed)

	# use softplus map to positive domain without exploding to massive values;
	# happens to be appropriate for the problems we work with here but might want something more configurable
	initial_params_array = jnp.array(list(initial_params.values()))
	to_param_dict = lambda d: dict(zip(initial_params.keys(), initial_params_array * softplus_norm(d)))

	# Initialize CMA-ES
	es = CMA_ES(
		population_size=pop_size,
		solution=initial_params_array * 0
	)

	# Initialize parameters and state
	params = es.default_params
	state = es.init(key, initial_params_array * 0, params)

	# number of domain-randomized rollouts per candidate evaluation
	samples = 10

	def evaluate(params: jnp.ndarray, key) -> float:
		"""Evaluate a candidate parameter vector over a randomized ensemble."""
		sim = replace_simulation(simulation, to_param_dict(params))
		keys = jax.random.split(key, samples)
		def c(k):
			m = sim.randomize_domain(k)
			return cost(m, m.run())

		costs = jax.jit(jax.vmap(c))(keys)
		# 90th percentile: optimize against a near-worst-case domain sample,
		# which is more robust than the mean and less brittle than the max.
		return jnp.percentile(costs, 90)


	# Optimization loop
	print(f"{'Generation':<10} {'Best Fitness':<15} {'Mean Fitness':<15}")
	print("-" * 40)

	@jax.jit
	def step(state, key):
		key, key_ask, key_eval, key_domain = jax.random.split(key, 4)

		# Generate a set of candidate solutions to evaluate
		population, state = es.ask(key_ask, state, params)

		# Evaluate the fitness of the population
		fitness = jax.vmap(lambda p: evaluate(p, key_domain))(population)

		# Update the evolution strategy
		state, metrics = es.tell(key_eval, population, fitness, state, params)
		return state, key

	# Ask-Eval-Tell loop
	for i in range(num_generations):
		state, key = step(state, key)

		# Print progress
		if i % 10 == 0:
			print(f"Iteration {i}, Lowest cost: {state.best_fitness:.4f}, Params: {state.best_solution}")

	best_params = to_param_dict(state.best_solution)
	return best_params, state.best_fitness
