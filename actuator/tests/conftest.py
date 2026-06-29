"""Pytest configuration for the actuator test suite.

This file makes pytest fixtures and command line options available to all test files
in this directory.
"""

from .pytest_utils import pytest_addoption, plot_action

# Re-export for pytest discovery
__all__ = ['pytest_addoption', 'plot_action']


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "visual: test that renders a plot / writes a PNG; deselect in CI with -m 'not visual'",
    )