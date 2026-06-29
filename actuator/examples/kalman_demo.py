"""Kalman filter demonstration.

Estimates position and velocity from noisy position measurements and
visualizes:
1. True vs estimated position with measurements and 2-sigma uncertainty
2. True vs estimated velocity
3. Estimation errors
4. Innovation sequence and normalized-innovation-squared consistency check
"""

from pathlib import Path

import jax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from actuator.kalman_observer import MotorObserver, create_kalman_params


def run_demo(show: bool = True, save_path=None) -> None:
    """Run the Kalman filter on a synthetic trajectory and plot the result."""
    # Simulation parameters
    dt = 0.0001  # 10 kHz
    sim_time = 2.0  # seconds
    t = np.arange(0, sim_time, dt)

    # A non-trivial trajectory: a frequency-modulated sine with a ramp partway through
    true_pos = 2 * np.sin(2 * np.pi * (0.5 * t + 0.2 * np.sin(2 * np.pi * 0.5 * t)))
    true_pos += np.minimum(0, t - sim_time / 2)
    true_vel = np.gradient(true_pos, dt)

    # Noisy encoder measurements
    encoder_resolution = 2 ** 16
    position_noise_std = 2 * np.pi / 2 ** 14
    measured_pos = true_pos + np.random.normal(0, position_noise_std, len(t))

    # Observer noise tuning. Position process noise is left small; velocity
    # process noise absorbs all unmodelled acceleration (no acceleration input).
    process_noise_pos = 1e-6
    process_noise_vel = 1e-2
    measurement_noise = position_noise_std ** 2

    kalman_params = create_kalman_params(
        dt=dt,
        process_noise_pos=process_noise_pos,
        process_noise_vel=process_noise_vel,
        measurement_noise=measurement_noise,
    )

    observer = MotorObserver(dt=dt, encoder_resolution=encoder_resolution)
    observer.params = kalman_params

    @jax.jit
    def step(state, inputs):
        state, outputs = observer.update(state, *inputs)
        return state, outputs

    encoder_count = (measured_pos * encoder_resolution / (2 * np.pi)).astype(np.int32)
    states, result = jax.lax.scan(step, init=observer.init_state(), xs=(encoder_count,))
    estimated_pos = result['position']
    estimated_vel = result['velocity']
    pos_std = result['position_uncertainty']
    vel_std = result['velocity_uncertainty']
    innovations = result['diagnostics'].get('innovation', 0)
    innovation_std = np.sqrt(result['diagnostics'].get('innovation_covariance', 0))

    # Position / velocity / error figure
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[2, 2, 1])

    ax_pos = fig.add_subplot(gs[0])
    ax_pos.plot(t, true_pos, 'g-', label='True Position', linewidth=2)
    ax_pos.plot(t, measured_pos, 'r.', markersize=2, alpha=0.3, label='Measurements')
    ax_pos.plot(t, estimated_pos, 'b-', label='Estimated Position', linewidth=1.5)
    ax_pos.fill_between(t, estimated_pos - 2 * pos_std, estimated_pos + 2 * pos_std,
                        color='blue', alpha=0.2, label='2σ Uncertainty')
    ax_pos.set_ylabel('Position (rad)')
    ax_pos.set_title('Kalman Filter Position Estimation')
    ax_pos.legend()
    ax_pos.grid(True)

    ax_vel = fig.add_subplot(gs[1], sharex=ax_pos)
    ax_vel.plot(t, true_vel, 'g-', label='True Velocity', linewidth=2)
    ax_vel.plot(t, estimated_vel, 'b-', label='Estimated Velocity', linewidth=1.5)
    ax_vel.fill_between(t, estimated_vel - 2 * vel_std, estimated_vel + 2 * vel_std,
                        color='blue', alpha=0.2, label='2σ Uncertainty')
    ax_vel.set_ylabel('Velocity (rad/s)')
    ax_vel.set_title('Kalman Filter Velocity Estimation')
    ax_vel.legend()
    ax_vel.grid(True)

    ax_err = fig.add_subplot(gs[2], sharex=ax_pos)
    ax_err.semilogy(t, np.abs(estimated_pos - true_pos), 'b-', label='Position Error (rad)', linewidth=1.5)
    ax_err.semilogy(t, np.abs(estimated_vel - true_vel), 'r-', label='Velocity Error (rad/s)', linewidth=1.5)
    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Absolute Error')
    ax_err.set_title('Estimation Error')
    ax_err.legend()
    ax_err.grid(True, which='both')

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)

    # Innovation consistency figure
    fig2 = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[2, 1])

    ax_innov = fig2.add_subplot(gs[0])
    ax_innov.plot(t, innovations, 'b-', label='Innovation', alpha=0.7)
    ax_innov.plot(t, 2 * innovation_std, 'r--', label='±2σ')
    ax_innov.plot(t, -2 * innovation_std, 'r--')
    ax_innov.fill_between(t, -2 * innovation_std, 2 * innovation_std, color='red', alpha=0.1)
    ax_innov.set_title('Innovation Sequence')
    ax_innov.set_ylabel('Innovation')
    ax_innov.legend()
    ax_innov.grid(True)

    normalized_innov_sq = (innovations / np.maximum(innovation_std, 1e-6)) ** 2
    ax_norm = fig2.add_subplot(gs[1], sharex=ax_innov)
    ax_norm.semilogy(t, normalized_innov_sq, 'b-', alpha=0.7)
    ax_norm.axhline(y=1.0, color='r', linestyle='--')
    ax_norm.set_title('Normalized Innovation Squared (should be ~1)')
    ax_norm.set_xlabel('Time (s)')
    ax_norm.set_ylabel('Normalized Innovation²')
    ax_norm.grid(True, which='both')

    plt.tight_layout()
    if show:
        plt.show()


def main() -> None:
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    run_demo(save_path=output_dir / 'kalman_demo.png')


if __name__ == "__main__":
    main()