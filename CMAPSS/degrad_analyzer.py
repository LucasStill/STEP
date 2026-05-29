import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import MinMaxScaler


class DegradationAnalyzer:
    def __init__(self, model, dataset, scaler, device, window_size, scaler_z, proto_size, progression_scaler=None, scale=False):
        """
        Initialize degradation analyzer

        Args:
            model: The trained model
            dataset: The dataset (train or test)
            scaler: Pre-fitted scaler for window data
            device: Computation device
            progression_scaler: Optional pre-fitted progression scaler from training data
        """
        self.model = model
        self.dataset = dataset
        self.scaler = scaler
        self.device = device
        self.progression_scaler = progression_scaler
        self.scale = scale
        self.window_size = window_size
        self.scaler_z = scaler_z
        self.proto_size = proto_size

    def _process_dataset(self, data, unit_id=None):
        """Process dataset for specific unit or all units"""
        if unit_id is not None:
            data = data[data[:, 0] == unit_id]
        return np.hstack((
            data[:, 0].reshape(-1, 1),
            data[:, 1].reshape(-1, 1),
            data[:, 5:]
        ))

    def _create_windows(self, data, window_size=30, step=1, threshold=80):
        """Create sliding windows from unit data"""
        windows = []
        rul_labels = []
        unique_units = np.unique(data[:, 0])

        for unit_id in unique_units:
            unit_mask = data[:, 0] == unit_id
            unit_data = data[unit_mask]
            n_samples = len(unit_data)

            max_cycle = np.max(unit_data[:, 1])
            unit_rul = max_cycle - unit_data[:, 1]

            for i in range(0, n_samples - window_size + 1, step):
                window = unit_data[i:i + window_size, 2:].flatten()
                target_rul = unit_rul[i + window_size - 1]

                if threshold < 0 or target_rul > threshold:
                    windows.append(window)
                    rul_labels.append(target_rul)

        return np.array(windows), np.array(rul_labels)

    def compute_progression_data(self, unit_id, window_size=None, method='angular'):
        """Compute progression data for a specific unit"""
        # scale=True scales the latent representations with the defined scaler

        if window_size is None:
            window_size = self.window_size

        # Process the selected unit
        sample = self._process_dataset(self.dataset, unit_id=unit_id)
        traj_windows, traj_rul = self._create_windows(sample, window_size=window_size, threshold=-1)

        if len(traj_windows) == 0:
            print(f"Warning: No windows created for unit {unit_id}")
            return None

        # Use the pre-fitted scaler for window data
        traj_windows_scaled = self.scaler.transform(traj_windows)

        # Get raw latent vectors
        self.model.eval()
        with torch.no_grad():
            traj_tensor = torch.FloatTensor(traj_windows_scaled).to(self.device)
            _, z_latent = self.model(traj_tensor)
            z_latent = z_latent.cpu().numpy()
            if self.scale:
                z_latent = self.scaler_z.transform(z_latent)

        # Choose progression method
        if method == 'angular':
            progression_scores = self.compute_angular_progression(z_latent)
        elif method == 'prototype_ratio':
            progression_scores = self.compute_prototype_ratio(z_latent)
        elif method == 'prototype_projection':
            progression_scores = self.compute_prototype_projection(z_latent)
        elif method == 'trajectory':
            start_point = z_latent[0]
            end_point = z_latent[-1]
            degradation_direction = end_point - start_point
            degradation_direction = degradation_direction / np.linalg.norm(degradation_direction)
            progression_scores = z_latent @ degradation_direction
        else:
            raise ValueError(f"Unknown method: {method}")

        # Store raw progression scores before normalization
        raw_progression_scores = progression_scores.copy()

        # Normalize progression scores if scaler is available
        if self.progression_scaler is not None:
            progression_scores = self.progression_scaler.transform(progression_scores.reshape(-1, 1)).flatten()
        else:
            # Local normalization as fallback
            progression_scores = (progression_scores - progression_scores.min()) / (
                        progression_scores.max() - progression_scores.min())

        # Calculate monotonicity metrics
        mono_progression = np.sum(np.diff(progression_scores) > 0) / len(np.diff(progression_scores))

        # Reconstruction quality check
        with torch.no_grad():
            recon_traj = self.model(traj_tensor)[0].cpu().numpy()
        reconstruction_error = np.mean(np.linalg.norm(traj_windows_scaled - recon_traj, axis=1))

        # Additional eventual HIs we could consider:
        angles_2d = np.arctan2(z_latent[:, 1], z_latent[:, 0])

        # NEW: Compute radial distances from origin (0,0)
        radii_2d = np.linalg.norm(z_latent[:, :2], axis=1)  # Distance from origin

        # Method 2: PCA to 2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_pca = pca.fit_transform(z_latent)
        angles_pca = np.arctan2(z_pca[:, 1], z_pca[:, 0])
        radii_pca = np.linalg.norm(z_pca, axis=1)  # Distance in PCA space

        # Method 3: Smart trajectory-based projection
        start_vector = z_latent[0]
        end_vector = z_latent[-1]
        progression_vector = end_vector - start_vector

        # Project all points onto this progression direction
        projections = z_latent @ progression_vector
        normalized_projections = (projections - projections.min()) / (projections.max() - projections.min())

        # Compute distance from the line connecting start and end points
        # This measures how much the trajectory deviates from a straight line
        line_deviations = []
        for point in z_latent:
            # Vector from start to current point
            v = point - start_vector
            # Vector along the progression direction
            u = progression_vector
            # Projection of v onto u
            proj = (np.dot(v, u) / np.dot(u, u)) * u
            # Deviation from the line (perpendicular distance)
            deviation = np.linalg.norm(v - proj)
            line_deviations.append(deviation)

        # quick metrics
        line_deviations = np.array(line_deviations)
        angle_diff = np.diff(angles_2d)
        monotonic_ratio = np.sum(angle_diff > 0) / len(angle_diff)
        health_indicator = (np.degrees(angles_2d) / 90.0) * radii_2d  # Scale angle to [0,1] then weight by radius
        radius_change = radii_2d[-1] - radii_2d[0]
        avg_deviation = np.mean(line_deviations)
        max_deviation = np.max(line_deviations)
        angle_radius_corr = np.corrcoef(angles_2d, radii_2d)[0, 1]

        results = {
            'z_latent': z_latent,
            'progression_scores': progression_scores,
            'raw_progression_scores': raw_progression_scores,  # Keep raw for reference
            'traj_rul': traj_rul,
            'monotonicity': {
                'progression': mono_progression,
                'rul': np.sum(np.diff(traj_rul) < 0) / len(np.diff(traj_rul))
            },
            'reconstruction_error': reconstruction_error,
            'trajectory_length': len(z_latent),
            'progression_method': method,
            'unit_id': unit_id,
            'is_normalized': self.progression_scaler is not None,

            # maybe also add the PCA metrics?
            # xx
            'angles': angles_2d,
            'radii': radii_2d,
            'angles_deg': np.degrees(angles_2d),
            'line_deviations': line_deviations,
            'HI_weighted': health_indicator,
            'metrics': {
                'monotonic_ratio': monotonic_ratio,
                'radius_change': radius_change,
                'avg_deviation': avg_deviation,
                'angle_radius_corr': angle_radius_corr
            }
        }

        return results

    def compute_all_progression_data(self, unit_ids=None, window_size=None, method='angular'):
        """Compute progression data for multiple units"""
        if unit_ids is None:
            unit_ids = np.unique(self.dataset[:, 0]).astype(int)

        if window_size is None:
            window_size = self.window_size

        progression_data = {}

        for unit_id in unit_ids:
            try:
                results = self.compute_progression_data(unit_id, window_size, method)
                if results is not None:
                    progression_data[unit_id] = results
                    #print(f"Computed progression data for unit {unit_id}")
                else:
                    print(f"Skipped unit {unit_id} (no data)")
            except Exception as e:
                print(f"Error processing unit {unit_id}: {e}")

        return progression_data

    def fit_progression_scaler(self, unit_ids=None, window_size=None, method='angular'):
        """
        Fit a progression scaler on the specified units
        Returns the fitted scaler for use with test data
        """
        print("Fitting progression scaler on training data...")

        if window_size is None:
            window_size = self.window_size

        all_progression_scores = []

        if unit_ids is None:
            unit_ids = np.unique(self.dataset[:, 0]).astype(int)

        for unit_id in unit_ids:
            try:
                # Compute progression data without normalization
                sample = self._process_dataset(self.dataset, unit_id=unit_id)
                traj_windows, _ = self._create_windows(sample, window_size=window_size, threshold=-1)

                if len(traj_windows) == 0:
                    continue

                traj_windows_scaled = self.scaler.transform(traj_windows)

                self.model.eval()
                with torch.no_grad():
                    traj_tensor = torch.FloatTensor(traj_windows_scaled).to(self.device)
                    _, z_latent = self.model(traj_tensor)
                    z_latent = z_latent.cpu().numpy()
                    if self.scale:
                        z_latent = self.scaler_z.transform(z_latent)

                # Compute raw progression scores
                if method == 'angular':
                    progression_scores = self.compute_angular_progression(z_latent)
                elif method == 'prototype_ratio':
                    progression_scores = self.compute_prototype_ratio(z_latent)
                elif method == 'prototype_projection':
                    progression_scores = self.compute_prototype_projection(z_latent)
                else:
                    start_point = z_latent[0]
                    end_point = z_latent[-1]
                    degradation_direction = end_point - start_point
                    degradation_direction = degradation_direction / np.linalg.norm(degradation_direction)
                    progression_scores = z_latent @ degradation_direction

                all_progression_scores.extend(progression_scores)
                #print(f"Unit {unit_id}: collected {len(progression_scores)} progression scores")

            except Exception as e:
                print(f"Error processing unit {unit_id} for scaler fitting: {e}")

        # Fit scaler on all progression scores
        if all_progression_scores:
            progression_scaler = MinMaxScaler(feature_range=(0, 1))
            progression_scaler.fit(np.array(all_progression_scores).reshape(-1, 1))
            self.progression_scaler = progression_scaler
            print(f"Fitted progression scaler on {len(all_progression_scores)} scores")
            print(f"Progression scaler's scale: {progression_scaler.scale_[0]:.4f}")
            return progression_scaler
        else:
            print("Warning: No progression scores collected for scaler fitting")
            return None

    # Keep the other methods (angular_progression, etc.) the same as before
    def compute_angular_progression(self, z_latent):
        """Compute progression based on angle from initial prototype"""
        initial_prototype = np.array([1.0] + [0.0] * (self.proto_size - 1), dtype=np.float64)
        failure_prototype = np.array([0.0, 1.0] + [0.0] * (self.proto_size - 2), dtype=np.float64)

        # Normalize latent vectors for angular calculations
        norms = np.linalg.norm(z_latent, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        z_normalized = z_latent / norms

        # Calculate angles from initial prototype with improved numerical stability
        dot_product = z_normalized @ initial_prototype
        # Clip to [-1, 1] with small epsilon for numerical stability
        dot_product = np.clip(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)

        with np.errstate(invalid='ignore'):  # Suppress warnings for invalid values
            angles = np.arccos(dot_product)

        # Handle any NaN values that might arise
        angles = np.nan_to_num(angles, nan=0.0)

        # Convert to progression score (0 to 1)
        progression_scores = angles / (np.pi / 2)

        return progression_scores

    def compute_prototype_ratio(self, z_latent):
        """Compute progression based on relative distances to prototypes"""
        initial_prototype = [1.0] + [0.0] * (self.proto_size - 1)
        failure_prototype = [0.0, 1.0] + [0.0] * (self.proto_size - 2)

        dist_to_initial = np.linalg.norm(z_latent - initial_prototype, axis=1)
        dist_to_failure = np.linalg.norm(z_latent - failure_prototype, axis=1)

        # Progression as relative closeness to failure prototype
        progression_scores = dist_to_initial / (dist_to_initial + dist_to_failure)

        return progression_scores

    def compute_prototype_projection(self, z_latent):
        """Project onto the learned degradation direction between prototypes"""
        initial_prototype = [1.0] + [0.0] * (self.proto_size - 1)
        failure_prototype = [0.0, 1.0] + [0.0] * (self.proto_size - 2)

        # Degradation direction is from initial to failure
        degradation_direction = failure_prototype - initial_prototype
        degradation_direction = degradation_direction / np.linalg.norm(degradation_direction)

        # Project all points onto this fixed direction
        progression_scores = z_latent @ degradation_direction

        # Normalize to [0, 1] range based on prototype positions
        initial_proj = initial_prototype @ degradation_direction
        failure_proj = failure_prototype @ degradation_direction
        progression_scores = (progression_scores - initial_proj) / (failure_proj - initial_proj)

        return progression_scores

    # Keep the smoothing methods the same
    def smooth_series(self, series, method='savgol', **kwargs):
        """Smooth a time series using various methods"""
        if method == 'savgol':
            window_length = kwargs.get('window_length', min(11, len(series) // 3 * 2 + 1))
            polyorder = kwargs.get('polyorder', 3)
            return signal.savgol_filter(series, window_length, polyorder)

        elif method == 'moving_avg':
            window = kwargs.get('window', 5)
            return np.convolve(series, np.ones(window) / window, mode='same')

        elif method == 'exponential':
            alpha = kwargs.get('alpha', 0.3)
            smoothed = np.zeros_like(series)
            smoothed[0] = series[0]
            for i in range(1, len(series)):
                smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i - 1]
            return smoothed

        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def smooth_all_progression_data(self, progression_data, method='savgol', **kwargs):
        """Apply smoothing to progression scores for all units"""
        smoothed_data = {}

        for unit_id, data in progression_data.items():
            # Use the normalized progression scores for smoothing
            original_scores = data['progression_scores']
            smoothed_scores = self.smooth_series(original_scores, method, **kwargs)

            smoothed_data[unit_id] = {
                'original': original_scores,
                'smoothed': smoothed_scores,
                'unit_id': unit_id
            }

        return smoothed_data

    def plot_progression_comparison(self, progression_data, smoothed_data=None, unit_ids=None):
        """Plot comparison between original and smoothed progression data"""
        if unit_ids is None:
            unit_ids = list(progression_data.keys())

        n_units = len(unit_ids)
        fig, axes = plt.subplots(n_units, 1, figsize=(12, 4 * n_units))
        if n_units == 1:
            axes = [axes]

        for idx, unit_id in enumerate(unit_ids):
            ax = axes[idx]
            original = progression_data[unit_id]['progression_scores']

            ax.plot(original, 'o-', alpha=0.7, label='Original', linewidth=1, markersize=3)

            if smoothed_data and unit_id in smoothed_data:
                smoothed = smoothed_data[unit_id]['smoothed']
                ax.plot(smoothed, '-', linewidth=2, label='Smoothed', color='red')

            ax.set_title(f'Unit {unit_id} - Progression Scores')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Progression Score')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def create_visualization(self, unit_id, window_size=None):
        """Create the full visualization for a specific unit (original plotting functionality)"""

        if window_size is None:
            window_size = self.window_size

        results = self.compute_progression_data(unit_id, window_size)

        z_latent = results['z_latent']
        progression_scores = results['progression_scores']
        z_pca = results['z_pca']
        traj_rul = results['traj_rul']

        # Smart trajectory-based projection (for orthogonal direction)
        start_point = z_latent[0]
        end_point = z_latent[-1]
        degradation_direction = end_point - start_point
        degradation_direction = degradation_direction / np.linalg.norm(degradation_direction)
        progression_scores_viz = z_latent @ degradation_direction
        residuals = z_latent - np.outer(progression_scores_viz, degradation_direction)
        ortho_scores = np.linalg.norm(residuals, axis=1)

        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: PCA projection
        scatter1 = ax1.scatter(z_pca[:, 0], z_pca[:, 1], c=np.arange(len(z_latent)),
                               cmap='viridis', s=50, edgecolor='k')
        ax1.plot(z_pca[:, 0], z_pca[:, 1], 'k-', alpha=0.3)
        ax1.set_xlabel('PCA Component 1')
        ax1.set_ylabel('PCA Component 2')
        ax1.set_title(f'Unit {unit_id} - PCA Projection')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Smart projection
        scatter2 = ax2.scatter(progression_scores_viz, ortho_scores, c=np.arange(len(z_latent)),
                               cmap='viridis', s=50, edgecolor='k')
        ax2.plot(progression_scores_viz, ortho_scores, 'k-', alpha=0.3)
        ax2.set_xlabel('Degradation Progression')
        ax2.set_ylabel('Orthogonal Variation')
        ax2.set_title(f'Unit {unit_id} - Smart Trajectory Projection')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Progression score over time
        progression_normalized = (progression_scores - progression_scores.min()) / (
                    progression_scores.max() - progression_scores.min())
        ax3.plot(progression_normalized, 'o-', linewidth=2, markersize=4)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Normalized Progression Score')
        ax3.set_title('Learned Degradation Progression')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Compare with RUL
        rul_normalized = (traj_rul - traj_rul.min()) / (traj_rul.max() - traj_rul.min())
        ax4.plot(progression_normalized, 'o-', label='Learned Progression', linewidth=2)
        ax4.plot(rul_normalized, 's-', label='Normalized RUL', alpha=0.7)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Normalized Value')
        ax4.set_title('Learned vs RUL-based Progression')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print metrics
        print(f"\n=== Unit {unit_id} Analysis ===")
        print(f"Trajectory length: {results['trajectory_length']} points")
        print(f"Monotonicity scores:")
        print(f"  Learned progression: {results['monotonicity']['progression']:.1%}")
        print(f"  PCA component 1: {results['monotonicity']['pca']:.1%}")
        print(f"  Raw RUL: {results['monotonicity']['rul']:.1%}")
        print(f"Average reconstruction error: {results['reconstruction_error']:.4f}")

        return results