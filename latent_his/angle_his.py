import torch

from latent_his.extract_latent import extract_latent_representations


def get_angles(model, window, latent_scale, device, scaler_z):
    model.eval()
    with torch.no_grad():
        z_latent = extract_latent_representations(model, window, device)

        # scale the latent if we want to, or keep it vanilla
        if latent_scale:
            z_latent = scaler_z.transform(z_latent)

    # Method 1: First two dimensions only
    angles_2d = np.arctan2(z_latent[:, 1], z_latent[:, 0])

    # Reference vector [1.0, 0.0] points along the positive x-axis
    reference_vector = np.array([1.0, 0.0])

    # Calculate the relative angle between each latent vector and the reference
    # This gives us the angular distance/difference
    relative_angles = []
    for i in range(len(z_latent)):
        latent_vector = z_latent[i, :2]  # Use first two dimensions

        # Calculate dot product and determinant for angle difference
        dot_product = reference_vector[0] * latent_vector[0] + reference_vector[1] * latent_vector[1]
        determinant = reference_vector[0] * latent_vector[1] - reference_vector[1] * latent_vector[0]

        # This gives the signed angle from reference_vector to latent_vector
        angle_diff = np.arctan2(determinant, dot_product)
        relative_angles.append(angle_diff)

    return np.array(relative_angles)


def get_angles_from_trajectory(model, traj_windows, scaler, scaler_z, latent_scale=False):
    traj_windows_scaled = scaler.transform(traj_windows) # timesteps, # params
    print(f"traj_windows_scaled: {traj_windows_scaled.shape}")
    angles = get_angles(model, traj_windows_scaled, latent_scale=latent_scale, scaler_z=scaler_z)
    #print(angles[:10])
    #angles = angles % (2 * np.pi) # just added for normalization test
    return angles

import matplotlib.pyplot as plt
import numpy as np

# Global alignment offset (computed once and reused)
GLOBAL_ALIGNMENT_OFFSET = None

def compute_global_alignment_offset(reference_trajectories, model, k=8, latent_scale=False, device='cpu', scaler_z=None):
    """
    Compute a global alignment offset from reference trajectories
    This should be called once with healthy/normal trajectories
    """
    global GLOBAL_ALIGNMENT_OFFSET

    all_initial_angles = []

    for traj in reference_trajectories:
        # Handle both single window and trajectory
        if len(traj.shape) == 1:
            traj = traj.reshape(1, -1)

        z_latent = extract_latent_representations(model, traj, device)

        if latent_scale:
            z_latent = scaler_z.transform(z_latent)

        # Get angles for first k windows (or all if trajectory is shorter)
        angles = np.arctan2(z_latent[:, 1], z_latent[:, 0])
        valid_k = min(k, len(angles))
        all_initial_angles.extend(angles[:valid_k])

    # Compute global reference (circular mean for angles)
    sin_mean = np.mean(np.sin(all_initial_angles))
    cos_mean = np.mean(np.cos(all_initial_angles))
    global_mean_angle = np.arctan2(sin_mean, cos_mean)

    GLOBAL_ALIGNMENT_OFFSET = -global_mean_angle
    print(f"Computed global alignment offset: {GLOBAL_ALIGNMENT_OFFSET:.6f} rad")

    return GLOBAL_ALIGNMENT_OFFSET

def compute_aligned_angles(z_latent, global_offset=None, mode='auto'):
    """
    Compute aligned angles for latent representations with proper phase handling

    Parameters:
    - z_latent: Single window (latent_dim,) or trajectory (n_windows, latent_dim)
    - global_offset: If None, uses the precomputed global offset
    - mode: 'auto', 'single', or 'trajectory'
    """
    if global_offset is None:
        if GLOBAL_ALIGNMENT_OFFSET is None:
            raise ValueError("No global offset computed. Call compute_global_alignment_offset first.")
        global_offset = GLOBAL_ALIGNMENT_OFFSET

    def normalize_angle(angle):
        """Normalize angle to [0, 2π) range"""
        return angle % (2 * np.pi)

    def ensure_positive_progression(angles):
        """Ensure angles progress positively by adding 2π when they drop"""
        if len(angles) <= 1:
            return angles

        # Make a copy to avoid modifying original
        result = angles.copy()

        for i in range(1, len(angles)):
            # If angle drops significantly (more than π), assume phase wrap
            if result[i] - result[i-1] < -np.pi:
                result[i] += 2 * np.pi
            # If we're still negative relative to start, add 2π
            elif result[i] < 0 and result[0] >= 0:
                result[i] += 2 * np.pi

        return result

    # Handle input shape and determine mode
    if len(z_latent.shape) == 1:
        # Single window
        raw_angle = np.arctan2(z_latent[1], z_latent[0])
        aligned_angle = raw_angle + global_offset
        # Normalize to [0, 2π) for consistent positive values
        aligned_angle = normalize_angle(aligned_angle)
        return aligned_angle
    else:
        # Multiple windows
        raw_angles = np.arctan2(z_latent[:, 1], z_latent[:, 0])

        # Auto-detect mode
        if mode == 'auto':
            mode = 'trajectory' if len(raw_angles) > 5 else 'single_windows'

        if mode == 'trajectory':
            # For trajectories: unwrap and ensure positive progression
            unwrapped_angles = np.unwrap(raw_angles)
            aligned_angles = unwrapped_angles + global_offset
            # Ensure positive progression
            aligned_angles = ensure_positive_progression(aligned_angles)
            return aligned_angles
        else:
            # For individual windows: normalize to [0, 2π)
            aligned_angles = raw_angles + global_offset
            aligned_angles = np.array([normalize_angle(angle) for angle in aligned_angles])
            return aligned_angles
def get_aligned_angles_for_data(data, model, global_offset=None, latent_scale=False, mode='auto', device='cpu', scaler_z=None):
    """
    Main function to get aligned angles for any input data

    Parameters:
    - data: Single window (features,) or trajectory (n_windows, features)
    - model: Your trained model
    - global_offset: Optional override for global offset
    - latent_scale: Whether to apply latent scaling
    - mode: 'auto', 'single', or 'trajectory'
    """
    model.eval()
    with torch.no_grad():
        # Extract latent representations
        z_latent = extract_latent_representations(model, data, device)

        if latent_scale:
            z_latent = scaler_z.transform(z_latent)

        # Compute aligned angles
        aligned_angles = compute_aligned_angles(z_latent, global_offset, mode)

        return aligned_angles

def plot_angular_evolution(traj_windows, aligned_angles, title="Angular Evolution"):
    """
    Plot the angular evolution for a trajectory
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Raw trajectory (first window as example)
    if len(traj_windows.shape) > 1:
        ax1.plot(traj_windows[0])
    else:
        ax1.plot(traj_windows)
    ax1.set_title('Input Time Series (First Window)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.grid(True)

    # Plot 2: Angular evolution
    ax2.plot(aligned_angles, 'b-', linewidth=2, label='Aligned Angle')
    ax2.axhline(y=0, color='r', linestyle='--', label='Reference (0 rad)')
    ax2.set_xlabel('Window Index')
    ax2.set_ylabel('Angle (radians)')
    ax2.set_title(title)
    ax2.legend()
    ax2.grid(True)

    # Add degrees on right axis
    ax2_deg = ax2.twinx()
    angles_deg = np.degrees(aligned_angles)
    ax2_deg.set_ylabel('Angle (degrees)')
    # Don't plot degrees line to avoid clutter, just set scale

    plt.tight_layout()
    plt.show()

    return fig