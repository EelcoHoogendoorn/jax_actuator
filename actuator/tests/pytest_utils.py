"""Shared pytest utilities for the actuator test suite.

This module provides common pytest fixtures and configuration for controlling
plot behavior across different test modules.
"""

import matplotlib.pyplot as plt
import pytest


def pytest_addoption(parser):
    """Add command line option for controlling plot behavior"""
    parser.addoption(
        "--close-plots",
        action="store_true",
        default=False,
        help="Close plots instead of showing them (for automated testing)"
    )


@pytest.fixture
def plot_action(request):
    """Fixture that returns the appropriate plot action based on command line flag
    
    Returns:
        plt.show: When running from IDE or without --close-plots flag
        plt.close: When running with --close-plots flag for automated testing
    """
    if request.config.getoption("--close-plots"):
        return plt.close
    else:
        return plt.show