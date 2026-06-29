"""General thermal simulation module for arbitrary thermal networks

Supports tridiagonal thermal systems with configurable nodes, resistances, and capacitances.
Useful for MOSFETs, motors, power electronics, etc.


"""


import jax
import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from actuator.jax_utils import register_dataclass



@jit
def _assemble_matrices(conductances, capacitances, power_mask, ambient_mask, T_ambient, dt):
    """JIT-compiled matrix assembly for thermal network
    
    Args:
        conductances: Array of conductance values [g12, g23, ..., g_ambient]
        capacitances: Array of capacitance values for each node
        power_mask: Boolean mask for nodes that receive power input
        ambient_mask: Boolean mask for nodes that couple to ambient
        T_ambient: Ambient temperature
        dt: Time step
        
    Returns:
        A_inv, M, B, T_ref: Matrices for implicit integration
    """
    n = len(capacitances)
    
    # Build conductance matrix for chain topology (hardcoded for performance)
    G = jnp.zeros((n, n))
    
    # Fill tridiagonal structure for chain connectivity
    for i in range(n - 1):
        g = conductances[i]  # Conductance between nodes i and i+1
        # Diagonal terms
        G = G.at[i, i].add(g)
        G = G.at[i+1, i+1].add(g)
        # Off-diagonal coupling
        G = G.at[i, i+1].set(-g)
        G = G.at[i+1, i].set(-g)
    
    # Add ambient coupling (last conductance value)
    g_ambient = conductances[-1]
    G = jnp.where(ambient_mask[:, None], 
                  G.at[:, :].add(jnp.outer(jnp.ones(n), jnp.eye(n)[n-1]) * g_ambient),
                  G)
    
    # Build capacitance matrix (diagonal)
    C = jnp.diag(capacitances)
    
    # Build input matrix B (power distribution)
    B = jnp.where(power_mask, 1.0, 0.0)
    
    # Reference temperature vector
    T_ref = jnp.where(ambient_mask, g_ambient * T_ambient, 0.0)
    
    # System matrices for implicit method
    A_impl = C + dt * G
    A_inv = jnp.linalg.inv(A_impl)
    M = A_inv @ C
    
    return A_inv, M, B, T_ref


@register_dataclass
@dataclass  
class Thermal:
    """General thermal network solver using two-phase approach
    
    Phase 1: Parse dict-based configuration into connectivity structure (once)
    Phase 2: JIT-compiled matrix assembly for specific parameter values (fast)
    """
    
    # Thermal resistances (K/W) - dict with keys like 'jc', 'ch', 'ha' 
    R: Dict[str, float]
    
    # Thermal capacitances (J/K) - dict with keys like 'j', 'c', 'h'
    C: Dict[str, float]
    
    # Node names in order (e.g., ['junction', 'case', 'heatsink'])
    nodes: List[str]
    
    # Ambient temperature (K)
    T_ambient: float = 298.0
    
    # Power input nodes (which nodes receive power input)
    power_nodes: List[str] = None
    
    def __post_init__(self):
        """Phase 1: Build connectivity structure for JIT compilation"""
        if self.power_nodes is None:
            self.power_nodes = [self.nodes[0]] if self.nodes else []
        
        # Validate configuration
        n_nodes = len(self.nodes)
        if n_nodes < 2:
            raise ValueError("Need at least 2 thermal nodes")
        
        # Build node index mapping
        self.node_idx = {node: i for i, node in enumerate(self.nodes)}
        
        # Extract conductance connections and build IJ array
        self._build_connectivity()
        
        # Validate that all required keys exist
        self._validate_keys()
    
    def _build_connectivity(self):
        """Build connectivity structure for JIT-compiled assembly"""
        n = len(self.nodes)
        
        # Build resistance connections (between adjacent nodes)
        self.resistance_pairs = []
        self.resistance_keys = []
        
        for i in range(n - 1):
            key = f"{self.nodes[i][0]}{self.nodes[i+1][0]}"
            self.resistance_pairs.append((i, i+1))
            self.resistance_keys.append(key)
        
        # Add ambient resistance for last node
        ambient_key = f"{self.nodes[-1][0]}a"
        self.resistance_keys.append(ambient_key)
        
        # Build IJ array for matrix assembly
        self.IJ = jnp.array(self.resistance_pairs)
        
        # Build power input indices
        self.power_indices = jnp.array([
            self.node_idx.get(node, -1) for node in self.power_nodes
        ])
        
        # Build ambient coupling indices (last node couples to ambient)
        self.ambient_indices = jnp.array([n-1] if n > 0 else [])
    
    def _validate_keys(self):
        """Validate that all required resistance and capacitance keys exist"""
        # Check resistance keys
        expected_R = set(self.resistance_keys)
        if set(self.R.keys()) != expected_R:
            raise ValueError(f"Expected R keys {expected_R}, got {set(self.R.keys())}")
        
        # Check capacitance keys
        expected_C = {node[0] for node in self.nodes}
        if set(self.C.keys()) != expected_C:
            raise ValueError(f"Expected C keys {expected_C}, got {set(self.C.keys())}")

    def build_matrices(self, dt: float):
        """Phase 2: JIT-compiled matrix assembly with current parameter values
        
        Returns:
            A_inv, M, B, T_ref: Matrices for implicit integration
        """
        # NOTE: this matrix assembly currently runs uncompiled at the start of
        #  every trajectory evaluation; it could be cached or jitted if it shows
        #  up as a bottleneck.
        # Extract conductance and capacitance values in correct order
        conductances = jnp.array([1.0 / self.R[key] for key in self.resistance_keys])
        capacitances = jnp.array([self.C[node[0]] for node in self.nodes])
        
        # Create boolean masks for power and ambient coupling
        n = len(self.nodes)
        power_mask = jnp.zeros(n, dtype=bool)
        for node in self.power_nodes:
            idx = self.node_idx.get(node, -1)
            if idx >= 0:
                power_mask = power_mask.at[idx].set(True)
        
        ambient_mask = jnp.zeros(n, dtype=bool)
        if n > 0:
            ambient_mask = ambient_mask.at[n-1].set(True)  # Last node couples to ambient
        
        # Use JIT-compiled assembly
        return _assemble_matrices(
            conductances, capacitances, power_mask, 
            ambient_mask, self.T_ambient, dt
        )

    # def step_function(self, dt: float) -> Callable:
    #     """Create a step function for time integration
    #
    #     Args:
    #         dt: Time step size
    #
    #     Returns:
    #         Function that takes (T_old, power) and returns T_new
    #     """
    #     A_inv, M, B, T_ref = self.build_matrices(dt)
    #
    #     def step_fn(T_old: jnp.ndarray, power: float) -> jnp.ndarray:
    #         """Single time step using implicit backward Euler
    #
    #         Args:
    #             T_old: Temperature state at previous time step
    #             power: Power input at current time step
    #
    #         Returns:
    #             T_new: Temperature state at current time step
    #         """
    #         # Affine part: b = A_inv @ (dt * (B * power + T_ref))
    #         b = A_inv @ (dt * (B * power + T_ref))
    #         # Linear transformation: T_new = M @ T_old + b
    #         T_new = M @ T_old + b
    #         return T_new
    #
    #     return step_fn


def simulate_thermal_network(initial_temps: jnp.ndarray,
                           power_profile: jnp.ndarray,
                           time_points: jnp.ndarray,
                           thermal: Thermal) -> jnp.ndarray:
    """Simulate thermal response of a general thermal network
    
    Args:
        initial_temps: Initial temperatures for each node
        power_profile: Power input at each time point  
        time_points: Time points for simulation
        thermal: Thermal network configuration
        
    Returns:
        Temperature history: shape (n_times, n_nodes)
    """
    dt = time_points[1] - time_points[0]
    A_inv, M, B, T_ref = thermal.build_matrices(dt)
    
    @jit
    def jitted_simulation(initial_temps, power_profile, A_inv, M, B, T_ref):
        def scan_fn(T_old, power):
            # Affine part: b = A_inv @ (dt * (B * power + T_ref))
            b = A_inv @ (dt * (B * power + T_ref))
            # Linear transformation: T_new = M @ T_old + b
            T_new = M @ T_old + b
            return T_new, T_new
        
        _, temps = jax.lax.scan(scan_fn, initial_temps, power_profile)
        return temps
    
    return jitted_simulation(initial_temps, power_profile, A_inv, M, B, T_ref)


def create_mosfet_thermal() -> Thermal:
    """Create a standard 3-node MOSFET thermal model"""
    return Thermal(
        R={'jc': 0.25, 'ch': 0.75, 'ha': 11.5},
        C={'j': 0.0002, 'c': 0.15, 'h': 50.0},
        nodes=['junction', 'case', 'heatsink'],
        T_ambient=298.0,
        power_nodes=['junction']
    )


def create_motor_thermal() -> Thermal:
    """Create a 4-node motor thermal model (windings -> rotor -> stator -> ambient)"""
    return Thermal(
        R={'wr': 0.1, 'rs': 0.8, 'sa': 2.5},  # winding-rotor, rotor-stator, stator-ambient
        C={'w': 0.05, 'r': 1.2, 's': 15.0},   # winding, rotor, stator capacitances
        nodes=['winding', 'rotor', 'stator'],
        T_ambient=298.0,
        power_nodes=['winding']
    )


def create_custom_thermal(n_nodes: int, 
                         R_values: List[float],
                         C_values: List[float],
                         T_ambient: float = 298.0) -> Thermal:
    """Create a custom n-node thermal network
    
    Args:
        n_nodes: Number of thermal nodes
        R_values: List of n-1 thermal resistances between adjacent nodes + 1 ambient resistance
        C_values: List of n thermal capacitances
        T_ambient: Ambient temperature
        
    Returns:
        Thermal network configuration
    """
    if len(R_values) != n_nodes:
        raise ValueError(f"Need {n_nodes} R values (including ambient), got {len(R_values)}")
    if len(C_values) != n_nodes:
        raise ValueError(f"Need {n_nodes} C values, got {len(C_values)}")
    
    # Generate simple single-letter node names for easier key generation
    if n_nodes <= 26:
        # Use letters: a, b, c, d, e, ...
        node_names = [chr(ord('a') + i) for i in range(n_nodes)]
    else:
        # Fall back to numbers for > 26 nodes
        node_names = [f"n{i}" for i in range(n_nodes)]
    
    # Build resistance dict
    R_dict = {}
    for i in range(n_nodes - 1):
        if n_nodes <= 26:
            key = f"{node_names[i]}{node_names[i+1]}"  # ab, bc, cd, etc.
        else:
            key = f"n{i}n{i+1}"  # n0n1, n1n2, etc.
        R_dict[key] = R_values[i]
    
    # Add ambient resistance for last node
    if n_nodes <= 26:
        R_dict[f"{node_names[-1]}a"] = R_values[-1]  # ea, fa, etc.
    else:
        R_dict[f"n{n_nodes-1}a"] = R_values[-1]
    
    # Build capacitance dict  
    if n_nodes <= 26:
        C_dict = {node_names[i]: C_values[i] for i in range(n_nodes)}
    else:
        C_dict = {f"n{i}": C_values[i] for i in range(n_nodes)}
    
    return Thermal(
        R=R_dict,
        C=C_dict, 
        nodes=node_names,
        T_ambient=T_ambient,
        power_nodes=[node_names[0]]  # Power to first node by default
    )

