"""Tests for the general thermal module.

To run tests with plots shown and live output (default when running from IDE):
    pytest actuator/tests/test_thermal.py -s

To run tests with plots closed automatically (for CI/automated testing):
    pytest actuator/tests/test_thermal.py --close-plots -s

The -s flag enables live output so you can see optimization progress.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from actuator.thermal import (
    Thermal, simulate_thermal_network, create_mosfet_thermal, 
    create_motor_thermal, create_custom_thermal
)


def simulate_mosfet_thermal_model():
    """Build the 3-node MOSFET thermal model and simulate a 50W pulse.

    Returns:
        (temps, t, mosfet) for reuse by tests and visualization.
    """
    mosfet = create_mosfet_thermal()
    t = jnp.linspace(0, 0.1, 1000)  # 100ms
    power = jnp.where(t < 0.02, 50.0, 0.0)  # 50W pulse for 20ms
    initial = jnp.array([298.0, 298.0, 298.0])  # All nodes start at ambient
    temps = simulate_thermal_network(initial, power, t, mosfet)
    return temps, t, mosfet


def test_mosfet_thermal_model():
    """Test 3-node MOSFET thermal model"""
    temps, t, mosfet = simulate_mosfet_thermal_model()
    assert temps.shape == (1000, 3), f"Expected shape (1000, 3), got {temps.shape}"
    assert jnp.all(temps >= 298.0), "All temperatures should be >= ambient"
    assert jnp.max(temps) < 500.0, "Maximum temperature should be reasonable"


def test_motor_thermal_model():
    """Test 3-node motor thermal model"""
    print("\n=== Testing Motor Thermal Model ===")
    
    motor = create_motor_thermal()
    print(f"Motor thermal model: {len(motor.nodes)} nodes")
    print(f"Nodes: {motor.nodes}")
    print(f"Resistances: {motor.R}")
    print(f"Capacitances: {motor.C}")
    
    # Test simulation
    t = jnp.linspace(0, 1.0, 1000)  # 1 second
    power = jnp.ones_like(t) * 100.0  # Constant 100W
    initial = jnp.array([298.0, 298.0, 298.0])
    
    temps = simulate_thermal_network(initial, power, t, motor)
    
    assert temps.shape == (1000, 3), f"Expected shape (1000, 3), got {temps.shape}"
    assert jnp.all(temps >= 298.0), "All temperatures should be >= ambient"

    print("Motor thermal simulation successful")


def test_custom_thermal_model():
    """Test custom n-node thermal model"""
    print("\n=== Testing Custom 5-Node Thermal Model ===")
    
    custom = create_custom_thermal(
        n_nodes=5,
        R_values=[0.1, 0.2, 0.3, 0.4, 5.0],  # 4 internal + 1 ambient
        C_values=[0.01, 0.05, 0.1, 0.2, 1.0]  # 5 capacitances
    )
    print(f"Custom thermal model: {len(custom.nodes)} nodes")
    print(f"Nodes: {custom.nodes}")
    print(f"R dict: {custom.R}")
    print(f"C dict: {custom.C}")
    
    # Test simulation
    t = jnp.linspace(0, 0.5, 500)
    power = jnp.ones_like(t) * 25.0  # 25W constant
    initial = jnp.array([298.0] * 5)  # All nodes at ambient
    
    temps = simulate_thermal_network(initial, power, t, custom)
    
    assert temps.shape == (500, 5), f"Expected shape (500, 5), got {temps.shape}"
    assert jnp.all(temps >= 298.0), "All temperatures should be >= ambient"

    print("Custom thermal simulation successful")


def test_thermal_visualization(plot_action):
    """Test thermal visualization"""
    print("\n=== Testing Thermal Visualization ===")
    
    temps, t, mosfet = simulate_mosfet_thermal_model()

    # Create and display plot
    plt.figure(figsize=(10, 6))
    for i, node in enumerate(mosfet.nodes):
        plt.plot(t * 1000, temps[:, i] - 273.15, label=node.capitalize())
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Temperature (°C)')
    plt.title('MOSFET Thermal Response to 50W Pulse')
    plt.legend()
    plt.grid(True)
    plot_action()  # Use fixture to either show or close
    
    print("Thermal visualization successful")


def test_thermal_validation():
    """Test thermal model validation"""
    print("\n=== Testing Thermal Model Validation ===")
    
    # Test that invalid configurations raise errors
    try:
        # Missing resistance
        invalid = Thermal(
            R={'jc': 1.0},  # Missing 'ch' and 'ha'
            C={'j': 0.001, 'c': 0.01, 'h': 1.0},
            nodes=['junction', 'case', 'heatsink']
        )
        assert False, "Should have raised ValueError for missing resistances"
    except ValueError as e:
        print(f"Correctly caught validation error: {e}")
    
    try:
        # Missing capacitance
        invalid = Thermal(
            R={'jc': 1.0, 'ch': 2.0, 'ha': 10.0},
            C={'j': 0.001},  # Missing 'c' and 'h'
            nodes=['junction', 'case', 'heatsink']
        )
        assert False, "Should have raised ValueError for missing capacitances"
    except ValueError as e:
        print(f"Correctly caught validation error: {e}")
    
    print("Thermal validation tests passed")


if __name__ == "__main__":
    test_mosfet_thermal_model()
    test_motor_thermal_model()
    test_custom_thermal_model()
    # For direct execution, use plt.show
    test_thermal_visualization(plt.show)
    test_thermal_validation()
    print("\n=== All thermal tests completed ===")
