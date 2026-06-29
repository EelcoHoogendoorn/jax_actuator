"""
Motor module for PMSM (Permanent Magnet Synchronous Motor) simulation.

This module provides a dataclass-based implementation of a PMSM motor model
that is compatible with JAX's functional programming model.
"""

from dataclasses import dataclass, replace
from typing import Tuple, Any, TypeVar, Type, Dict

import jax
import jax.numpy as jnp
from jax import config

# Enable float64 for better numerical stability
config.update("jax_enable_x64", True)

# Type alias for array-like types that can be converted to JAX arrays
ArrayLike = jax.typing.ArrayLike

from .jax_utils import register_dataclass


@register_dataclass
@dataclass
class MotorState:
    """State variables for the PMSM motor model."""
    # Electrical state (dq reference frame)
    current_d: float = 0.0  # d-axis current (A)
    current_q: float = 0.0  # q-axis current (A)

    # Mechanical state
    position: float = 0.0  # Rotor position (rad)
    velocity: float = 0.0  # Rotor velocity (rad/s)

    # Thermal state (placeholder for future implementation)
    temperature: float = 25.0  # Temperature (°C)


@register_dataclass
@dataclass
class Motor:
    """Permanent Magnet Synchronous Motor (PMSM) model.
    
    All parameters are in SI units.

    NOTE: maximum current and voltage limits are properties of the controller, not the motor.
    The motor's electrical parameters (resistance, inductance, etc.) are physical
    properties, while current/voltage limits are determined by the controller's
    power electronics and cooling system.
    """
    # Electrical parameters
    resistance: float  # Stator resistance (ohm)
    inductance_d: float  # d-axis inductance (H)
    inductance_q: float  # q-axis inductance (H)
    Kt: float
    pole_pairs: int  # Number of pole pairs
    slots: int
    
    # Mechanical parameters
    inertia: float  # Rotor moment of inertia (kg·m²)
    friction: float  # Viscous friction coefficient (N·m·s/rad)
    hysteresis: float
    torque_static: float

    dedent_offset: float = jnp.pi*2 # to be multiplied by 0-1

    @property
    def Lq(self):
        """conversion from dq to phase-neutral"""
        return self.inductance_q * (2/3)
    @property
    def Ld(self):
        """conversion from dq to phase-neutral"""
        return self.inductance_d * (2/3)
    @property
    def R(self):
        """conversion from dq to phase-neutral"""
        return self.resistance * (2/3)


    @property
    def flux_linkage(self) -> float:
        """conversion from dq to phase-neutral"""
        return self.Kt / self.pole_pairs / (3/2)

    @classmethod
    def create_default(cls) -> 'Motor':
        """Create a motor with default parameters similar to the original implementation."""
        pole_pairs = 16
        Kt = 1.2  # Torque constant (N·m/A)
        
        return cls(
            resistance=0.24,  # ohm
            inductance_d=0.0002,  # H
            inductance_q=0.0003,  # H
            Kt=Kt,
            pole_pairs=pole_pairs,
            slots=36,
            inertia=1e-1,  # kg·m²
            friction=0.001,  # N·m·s/rad
            hysteresis=0.0,
            torque_static=0.0,
            dedent_offset=0.0,
        )


    def voltage_feedforward(
        self,
        i_d: float,
        i_q: float,
        di_dt_d: float,
        di_dt_q: float,
        velocity_mech: float
    ) -> Tuple[float, float]:
        """Calculate feedforward voltage commands based on current state and references.

        Implements the PMSM voltage equations:
            Vd = R*id + Ld*did/dt - ω*Lq*iq
            Vq = R*iq + Lq*diq/dt + ω*(Ld*id + λ)

        Args:
            i_d: D-axis current (A)
            i_q: Q-axis current (A)
            di_dt_d: Desired rate of change of d-axis current (A/s)
            di_dt_q: Desired rate of change of q-axis current (A/s)
            velocity_mech: Mechanical velocity (rad/s)
            
        Returns:
            Tuple of (Vd, Vq) feedforward voltages (V)
        """
        omega_elec = velocity_mech * self.pole_pairs
        
        vd = (
            self.R * i_d +
            self.Ld * di_dt_d -
            omega_elec * self.Lq * i_q
        )
        
        vq = (
            self.R * i_q +
            self.Lq * di_dt_q +
            omega_elec * (self.Ld * i_d + self.flux_linkage)
        )
        
        return vd, vq

    def current_feedforward(
        self,
        v_d: float,
        v_q: float,
        i_d: float,
        i_q: float,
        velocity_mech: float
    ) -> Tuple[float, float]:
        """Calculate current derivatives based on voltage commands and current state.
        
        Solves the PMSM voltage equations for did/dt and diq/dt:
            did/dt = (Vd - R*id + ω*Lq*iq) / Ld
            diq/dt = (Vq - R*iq - ω*(Ld*id + λ)) / Lq
        
        Args:
            v_d: D-axis voltage (V)
            v_q: Q-axis voltage (V)
            i_d: D-axis current (A)
            i_q: Q-axis current (A)
            velocity_mech: Mechanical velocity (rad/s)
            
        Returns:
            Tuple of (did/dt, diq/dt) current derivatives (A/s)
        """
        omega_elec = velocity_mech * self.pole_pairs
        
        # Note: The back-EMF terms are subtracted from the voltage terms
        di_dt_d = (
            v_d -
            self.R * i_d -
            (-omega_elec * self.Lq * i_q)  # Back-EMF d-axis
        ) / self.Ld
        
        di_dt_q = (
            v_q -
            self.R * i_q -
            (omega_elec * (self.Ld * i_d + self.flux_linkage))  # Back-EMF q-axis
        ) / self.Lq
        
        return di_dt_d, di_dt_q



    def update_electrical(
        self,
        state: 'MotorState',
        voltage_d: float,
        voltage_q: float,
        dt: float
    ) -> 'MotorState':
        """Update the electrical state of the motor.

        Implements the electrical dynamics:
            did/dt = (Vd - R*id + ω*Lq*iq) / Ld
            diq/dt = (Vq - R*iq - ω*(Ld*id + λ)) / Lq

        Uses Euler integration to update the currents.

        Args:
            state: Current motor state
            voltage_d: D-axis voltage (V)
            voltage_q: Q-axis voltage (V)
            dt: Time step (s)

        Returns:
            Updated motor state with new current values
        """
        # Calculate current derivatives
        di_dt_d, di_dt_q = self.current_feedforward(
            voltage_d, voltage_q,
            state.current_d, state.current_q,
            state.velocity
        )

        # Update currents using Euler integration
        new_current_d = state.current_d + di_dt_d * dt
        new_current_q = state.current_q + di_dt_q * dt

        return replace(state,
            current_d=new_current_d,
            current_q=new_current_q
        )

    def dedent(self, aa):
        """compute dedent torque term"""
        aa = aa + self.dedent_offset
        poles = self.pole_pairs * 2
        slots = self.slots
        # slots = poles - 4   # works for 40-36 config
        fp = [1 / 2, 1, 2]  # this seems decent based on odrive example
        fs = [1, 2, 3]
        h = lambda f, a: jnp.sin(aa * f) * a
        return sum(h(poles * q, 0.8) for q in fp) + sum(h(slots * q, 0.8) for q in fs)

    def saturation_factor(self, amps):
        """This saturation curve is based on the following rule of thumb:
		saturation-wise its generally possible [1] to get double the torque one gets at the end of the linear range,
		but obtaining that doubled torque requires some 3x the current, rather than 2x;
		that is the average Kt beyond the linear range is roughly cut in half
		Beyond that range things might get more nonlinear still;
		given the absence of empirical data on real motors in that regime,
		going deeper than 2x beyond the linear torque range should be taken with a particularly big grain of salt.

		[1] as judged by a few outrunners for which such data is available
		"""
        saturation = 80
        softmaxout = lambda x, y, h: jnp.log(jnp.exp(x * h) + jnp.exp(y * h)) / h
        return softmaxout(1, (jnp.abs(amps) / saturation - 1) / 4 + 1, 20)

    def torque(self, state: MotorState):
        """motor output torque estimate from current, with saturation model applied.

        Includes the alignment term (flux_linkage * iq) and the reluctance term
        ((Ld - Lq) * id * iq); the reluctance scaling is approximate.
        """
        return 1.5 * self.pole_pairs * (
            self.flux_linkage * state.current_q +
            (self.Ld - self.Lq) * state.current_d * state.current_q
        ) / self.saturation_factor(state.current_q)

    def update_mechanical(
        self,
        state: 'MotorState',
        load_torque: float,
        dt: float
    ) -> 'MotorState':
        """Update the mechanical state of the motor.
        
        Args:
            state: Current motor state
            load_torque: External load torque (N·m)
            dt: Time step (s)
            
        Returns:
            Updated motor state with new mechanical state
        """
        # Electromagnetic torque (simplified)
        torque = self.torque(state)


        # Total torque (electrical - friction - load)
        friction = self.friction * state.velocity + self.hysteresis * jnp.sign(state.velocity)
        total_torque = torque - friction - load_torque + self.dedent(state.position)
        
        # Update velocity and position using Euler integration
        acceleration = total_torque / self.inertia
        new_velocity = state.velocity + acceleration * dt
        # stick velocity to zero if torque required is less than torque_static
        is_dynamic = jnp.abs(new_velocity) * self.inertia > self.torque_static * dt
        new_velocity = new_velocity * is_dynamic
        new_position = state.position + new_velocity * dt
        
        # Normalize position to [0, 2π)
        new_position = jnp.mod(new_position, 2 * jnp.pi)
        
        return replace(state,
            position=new_position,
            velocity=new_velocity
        )
    
    def update(
        self,
        state: 'MotorState',
        voltage_d: float,
        voltage_q: float,
        load_torque: float,
        dt: float,
        substeps=4,
    ) -> 'MotorState':
        """Update the complete motor state for one time step.
        
        Args:
            state: Current motor state
            voltage_d: d-axis voltage (V)
            voltage_q: q-axis voltage (V)
            load_torque: External load torque (N·m)
            dt: Time step (s)
            substeps: Number of substeps to use for numerical integration
                physical logic should be more accurate than controller ff;
                want to be able to simulate such discrepancies
            
        Returns:
            Updated motor state
        """
        def substep(state, voltage_d, voltage_q, load_torque, dt):
            state = self.update_electrical(state, voltage_d, voltage_q, dt)
            return self.update_mechanical(state, load_torque, dt)

        state = jax.lax.fori_loop(
            0, substeps,
            lambda i, state: substep(state, voltage_d, voltage_q, load_torque, dt / substeps),
            state
        )

        return state