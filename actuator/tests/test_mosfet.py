"""Tests for the MOSFET thermal module.

To run tests with plots shown and live output (default when running from IDE):
    pytest actuator/tests/test_mosfet.py -s

To run tests with plots closed automatically (for CI/automated testing):
    pytest actuator/tests/test_mosfet.py --close-plots -s

The -s flag enables live output so you can see optimization progress.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from actuator.mosfet import MOSFETThermal, simulate_mosfet_thermal, tune_mosfet_parameters, pulse_response


def simulate_basic_mosfet():
    """Simulate a 50W pulse through the MOSFET thermal model.

    Returns:
        (temps, t) for reuse by tests and visualization.
    """
    mosfet = MOSFETThermal()
    t = jnp.linspace(0, 0.1, 1000)  # 100ms
    power = jnp.where(t < 0.02, 50.0, 0.0)  # 50W pulse for 20ms
    initial = jnp.array([298.0, 298.0, 298.0])  # All nodes start at ambient
    temps = simulate_mosfet_thermal(initial, power, t, mosfet)
    return temps, t


def test_mosfet_basic_simulation():
    """Test basic MOSFET thermal simulation"""
    temps, t = simulate_basic_mosfet()
    T_j_final = temps[-1, 0]
    assert T_j_final > 298.0, "Junction temperature should increase"
    assert T_j_final < 400.0, "Junction temperature should be reasonable"


def test_mosfet_pulse_response():
    """Test MOSFET pulse response visualization"""
    print("\n=== Testing MOSFET Pulse Response ===")
    
    mosfet = MOSFETThermal()
    
    # Test pulse response data generation (without showing plot)
    import jax.numpy as jnp
    t = jnp.linspace(0, 0.1, int(0.1 * 10000))  
    power = jnp.where(t < 0.02, 50.0, 0.0)
    initial = jnp.array([mosfet.T_ambient] * 3)
    
    temps = simulate_mosfet_thermal(initial, power, t, mosfet)
    
    # Basic validation
    assert temps.shape[0] == len(t), "Temperature array should match time array"
    assert temps.shape[1] == 3, "Should have 3 thermal nodes"
    assert jnp.max(temps[:, 0]) > mosfet.T_ambient, "Junction should heat up"
    
    print("Pulse response data generated successfully")


def test_mosfet_parameter_tuning():
    """Test MOSFET parameter tuning with reduced target points"""
    print("\n=== Testing MOSFET Parameter Tuning ===")
    
    # Use fewer target points for faster testing
    target_points = [(0.01, 120.0), (0.1, 40.0)]
    tuned_mosfet = tune_mosfet_parameters(target_points, verbose=True)
    
    # Verify tuned parameters are reasonable
    assert tuned_mosfet.R_jc > 0, "R_jc should be positive"
    assert tuned_mosfet.R_ch > 0, "R_ch should be positive" 
    assert tuned_mosfet.R_ha > 0, "R_ha should be positive"
    assert tuned_mosfet.C_j > 0, "C_j should be positive"
    assert tuned_mosfet.C_c > 0, "C_c should be positive"
    assert tuned_mosfet.C_h > 0, "C_h should be positive"
    
    print("Parameter tuning completed successfully")


def test_mosfet_thermal_plot(plot_action):
    """Test thermal response plotting"""
    print("\n=== Testing Thermal Response Plot ===")
    
    temps, t = simulate_basic_mosfet()

    # Create and display plot
    plt.figure(figsize=(10, 6))
    for i, node in enumerate(['Junction', 'Case', 'Heatsink']):
        plt.plot(t * 1000, temps[:, i] - 273.15, label=node)
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Temperature (°C)')
    plt.title('MOSFET Thermal Response')
    plt.legend()
    plt.grid(True)
    plot_action()  # Use fixture to either show or close
    
    print("Thermal response plot generated successfully")


if __name__ == "__main__":
    test_mosfet_basic_simulation()
    test_mosfet_pulse_response()
    test_mosfet_parameter_tuning()
    
    # For direct execution, use plt.show
    test_mosfet_thermal_plot(plt.show)
    print("\n=== All MOSFET tests completed ===")
