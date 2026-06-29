"""Tests for JAX utility functions."""

import jax.numpy as jnp
import pytest
from dataclasses import dataclass

from actuator.jax_utils import register_dataclass, replace_pytree_values


@register_dataclass
@dataclass
class Inner:
    d: jnp.ndarray
    e: list


@register_dataclass
@dataclass
class Outer:
    b: jnp.ndarray
    c: Inner


def test_replace_pytree_values():
    """Test that values in a pytree can be replaced using path strings."""
    # Example pytree with dataclass
    pytree = {
        'a': Outer(
            b=jnp.array(1),
            c=Inner(
                d=jnp.array(2),
                e=[jnp.array(3), jnp.array(4)]
            )
        ),
        'f': jnp.array(5)
    }

    # Dictionary with replacement values
    replace_dict = {
        'a__b': jnp.array(10),  # Dataclass attribute
        'a__c__d': jnp.array(20),  # Nested dataclass attribute
        'a__c__e__0': jnp.array(30),  # List inside dataclass
        'f': jnp.array(50)  # Top-level attribute
    }

    # Replace values
    new_pytree = replace_pytree_values(pytree, replace_dict)

    # Check that the values were replaced correctly
    assert new_pytree['a'].b == 10
    assert new_pytree['a'].c.d == 20
    assert new_pytree['a'].c.e[0] == 30
    assert new_pytree['f'] == 50
    
    # Check that non-replaced values remain the same
    assert new_pytree['a'].c.e[1] == 4


def test_replace_pytree_values_nonexistent_path():
    """Test that non-existent paths in replacements are ignored."""
    pytree = {'a': jnp.array(1), 'b': jnp.array(2)}
    
    # Try to replace a non-existent path
    new_pytree = replace_pytree_values(
        pytree,
        {'nonexistent': jnp.array(99), 'a': jnp.array(10)}
    )
    
    # Check that only the existing path was replaced
    assert new_pytree['a'] == 10
    assert new_pytree['b'] == 2  # Unchanged


def test_replace_pytree_values_nested_dict():
    """Test replacement in nested dictionaries."""
    pytree = {
        'a': {
            'b': jnp.array(1),
            'c': [jnp.array(2), jnp.array(3)]
        },
        'd': jnp.array(4)
    }
    
    new_pytree = replace_pytree_values(
        pytree,
        {
            'a__b': jnp.array(10),
            'a__c__1': jnp.array(30),
            'd': jnp.array(40)
        }
    )
    
    assert new_pytree['a']['b'] == 10
    assert new_pytree['a']['c'][0] == 2  # Unchanged
    assert new_pytree['a']['c'][1] == 30
    assert new_pytree['d'] == 40
