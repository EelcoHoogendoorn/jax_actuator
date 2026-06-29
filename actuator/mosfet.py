"""MOSFET thermal model using the general thermal simulation framework

Provides MOSFET-specific thermal analysis and parameter tuning.
"""

import jax
import jax.numpy as jnp
from jax import jit, grad
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple
from actuator.jax_utils import register_dataclass
from actuator.thermal import Thermal, simulate_thermal_network


@register_dataclass
@dataclass
class MOSFETThermal:
    """MOSFET thermal model with parameter tuning capabilities"""
    
    # Thermal resistances (K/W)
    R_jc: float = 1.0    # Junction to case
    R_ch: float = 1.75   # Case to heatsink (includes TIM)
    R_ha: float = 30.5   # Heatsink to ambient

    # Thermal capacitances (J/K)
    C_j: float = 4e-3    # Junction
    C_c: float = 2.7e-2  # Case/bulk
    C_h: float = 2e-0    # Heatsink

    # Ambient temperature (K)
    T_ambient: float = 298.0  # 25°C
    
    def to_thermal(self) -> Thermal:
        """Convert to general Thermal object"""
        return Thermal(
            R={'jc': self.R_jc, 'ch': self.R_ch, 'ha': self.R_ha},
            C={'j': self.C_j, 'c': self.C_c, 'h': self.C_h},
            nodes=['junction', 'case', 'heatsink'],
            T_ambient=self.T_ambient,
            power_nodes=['junction']
        )


def simulate_mosfet_thermal(initial_temps: jnp.ndarray,
                           power_profile: jnp.ndarray,
                           time_points: jnp.ndarray,
                           mosfet: MOSFETThermal) -> jnp.ndarray:
    """Simulate MOSFET thermal response
    
    Args:
        initial_temps: Initial temperatures [T_j, T_c, T_h]
        power_profile: Power dissipation at each time point (W)
        time_points: Time points for simulation
        mosfet: MOSFET thermal parameters

    Returns:
        Temperature history: shape (n_times, 3)
    """
    thermal = mosfet.to_thermal()
    return simulate_thermal_network(initial_temps, power_profile, time_points, thermal)


def tune_mosfet_parameters(target_points: list, T_j_max: float = 423.0, verbose: bool = True) -> MOSFETThermal:
    """
    Advanced MOSFET parameter tuning using multiple target points.

    Args:
        target_points: List of (duration, power) tuples
        T_j_max: Maximum junction temperature (K)
        verbose: Print optimization progress

    Returns:
        MOSFETThermal: Tuned parameters
    """
    if verbose:
        print(f"Tuning MOSFET parameters for {len(target_points)} target points:")
        for duration, power in target_points:
            print(f"  {power}W @ {duration * 1000:.1f}ms")

    # Define the multi-point objective function with multiplicative scaling
    def multi_objective(scale_factors):
        # Use multiplicative scaling: param_new = param_init * exp(scale_factor)
        # This keeps parameters positive and allows gradient descent on similar scales
        scale_factors = jnp.clip(scale_factors, -3.0, 3.0)

        R_jc = R_jc_init * jnp.exp(scale_factors[0])
        R_ch = R_ch_init * jnp.exp(scale_factors[1])
        R_ha = R_ha_init * jnp.exp(scale_factors[2])
        C_j = C_j_init * jnp.exp(scale_factors[3])
        C_c = C_c_init * jnp.exp(scale_factors[4])
        C_h = C_h_init * jnp.exp(scale_factors[5])

        mosfet = MOSFETThermal(R_jc=R_jc, R_ch=R_ch, R_ha=R_ha,
                              C_j=C_j, C_c=C_c, C_h=C_h)

        total_error = 0.0

        try:
            for duration, target_power in target_points:
                # Adaptive time resolution but with limits
                n_points = min(max(int(duration * 1000), 10), 200)
                t = jnp.linspace(0, duration, n_points)
                power = jnp.ones_like(t) * target_power
                initial = jnp.array([298.0, 298.0, 298.0])

                temps = simulate_mosfet_thermal(initial, power, t, mosfet)
                T_j_peak = jnp.max(temps[:, 0])

                # Check for numerical issues
                if jnp.isnan(T_j_peak) or jnp.isinf(T_j_peak):
                    return 1e6

                # Relative error
                error = ((T_j_peak - T_j_max) / T_j_max) ** 2
                total_error += error

            return total_error

        except:
            return 1e6

    default = MOSFETThermal()
    R_jc_init = default.R_jc
    R_ch_init = default.R_ch
    R_ha_init = default.R_ha

    C_j_init = default.C_j
    C_c_init = default.C_c
    C_h_init = default.C_h
    # Start optimization from zero scaling (i.e., initial parameters)
    scale_factors_0 = jnp.zeros(6)

    # Optimize with simple gradient descent on scale factors
    if verbose:
        print("Optimizing...")

    scale_factors = scale_factors_0
    best_scale_factors = scale_factors
    best_loss = float('inf')
    learning_rate = 0.5  # Can use larger LR since all parameters are on similar scale now

    for i in range(100):
        try:
            g = grad(multi_objective)(scale_factors)
            # Clip gradients
            g = jnp.clip(g, -1.0, 1.0)
            scale_factors = scale_factors - learning_rate * g

            loss = multi_objective(scale_factors)
            if loss < best_loss:
                best_loss = loss
                best_scale_factors = scale_factors

            if verbose and i % 20 == 0:
                print(f"  Step {i}: loss = {loss:.6f}")
        except:
            if verbose:
                print(f"  Step {i}: optimization failed, using best so far")
            break

    # Extract final parameters using the best scale factors
    final_scale_factors = best_scale_factors
    final_scale_factors = jnp.clip(final_scale_factors, -3.0, 3.0)
    R_jc = float(R_jc_init * jnp.exp(final_scale_factors[0]))
    R_ch = float(R_ch_init * jnp.exp(final_scale_factors[1]))
    R_ha = float(R_ha_init * jnp.exp(final_scale_factors[2]))
    C_j = float(C_j_init * jnp.exp(final_scale_factors[3]))
    C_c = float(C_c_init * jnp.exp(final_scale_factors[4]))
    C_h = float(C_h_init * jnp.exp(final_scale_factors[5]))

    mosfet = MOSFETThermal(
        R_jc=R_jc, R_ch=R_ch, R_ha=R_ha,
        C_j=C_j, C_c=C_c, C_h=C_h
    )

    if verbose:
        print(f"\nFinal parameters:")
        print(f"  R_jc = {R_jc:.3f} K/W")
        print(f"  R_ch = {R_ch:.3f} K/W")
        print(f"  R_ha = {R_ha:.3f} K/W")
        print(f"  Total R = {R_jc + R_ch + R_ha:.3f} K/W")
        print(f"  C_j = {C_j * 1000:.3f} mJ/K")
        print(f"  C_c = {C_c:.3f} J/K")
        print(f"  C_h = {C_h:.1f} J/K")

        # Verify the fit
        print(f"\nVerification:")
        for duration, target_power in target_points:
            n_points = min(int(duration * 10000), 1000)
            t = jnp.linspace(0, duration, n_points)
            power = jnp.ones_like(t) * target_power
            initial = jnp.array([298.0, 298.0, 298.0])

            temps = simulate_mosfet_thermal(initial, power, t, mosfet)
            T_j_peak = jnp.max(temps[:, 0])

            print(
                f"  {duration * 1000:.1f}ms @ {target_power}W: T_j = {T_j_peak - 273.15:.1f}°C (target: {T_j_max - 273.15:.1f}°C)")

    return mosfet


def pulse_response(mosfet: MOSFETThermal, power_level: float = 50.0, pulse_duration: float = 0.02, total_time: float = 0.1):
    """Plot thermal response to a power pulse
    
    Args:
        mosfet: MOSFET thermal model
        power_level: Power level in watts
        pulse_duration: Duration of pulse in seconds
        total_time: Total simulation time in seconds
    """
    # Simulate a pulse response
    t = jnp.linspace(0, total_time, int(total_time * 10000))  
    power = jnp.where(t < pulse_duration, power_level, 0.0)
    initial = jnp.array([mosfet.T_ambient] * 3)

    temps = simulate_mosfet_thermal(initial, power, t, mosfet)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Temperature plot
    ax1.plot(t * 1000, temps[:, 0] - 273.15, label='Junction')
    ax1.plot(t * 1000, temps[:, 1] - 273.15, label='Case')
    ax1.plot(t * 1000, temps[:, 2] - 273.15, label='Heatsink')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'Thermal Response to {power_level}W Pulse')

    # Power plot
    ax2.plot(t * 1000, power)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Power (W)')
    ax2.grid(True)
    ax2.set_title('Power Profile')

    plt.tight_layout()
    plt.show()


