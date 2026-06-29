"""Utilities for working with JAX and dataclasses."""

import jax
import jax.tree_util
from typing import Any, Dict


# Register dataclasses as PyTree nodes by default
register_dataclass = jax.tree_util.register_dataclass


def replace_pytree_values(pytree: Any, replacements: Dict[str, Any]) -> Any:
    """Replace values in a JAX pytree using a dictionary of replacements.
    
    Args:
        pytree: A JAX PyTree (nested structure of arrays, dicts, lists, tuples, etc.)
        replacements: Dictionary mapping paths to replacement values. Paths are __-separated
                     strings representing the path to each value in the pytree.
                     A leading __ signals a wildcard suffix that matches any
                     preceding prefix path. Values may be plain replacements or
                     callables applied to the existing value.
    Returns:
        A new pytree with the specified values replaced.
    """
    def get(path, value):
        val = replacements[path]
        try:
            return val(value)
        except:
            return val

    def replace_fn_wildcard(path, value):
        path_str = ''
        for c in reversed(path):
            path_str = str(getattr(c, 'key', getattr(c, 'name', getattr(c, 'idx', str(c))))) + path_str
            if path_str in replacements:
                return get(path_str, value)
            path_str = '__' + path_str
            if path_str in replacements:
                return get(path_str, value)
        return value
    return jax.tree_util.tree_map_with_path(replace_fn_wildcard, pytree)


def lerp_pytree(tree1, tree2, t):
    return jax.tree_util.tree_map(lambda x, y: (1 - t) * x + t * y, tree1, tree2)
