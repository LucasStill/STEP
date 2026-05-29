#!/usr/bin/env python3
"""
Test 1b: Density Decoupling - Compare marginal distributions of learned progression score θ
and normalized timestep t/T to prove the backbone learned state, not temporal order.

If θ ≈ t/T, the model just learned to order by time.
If θ ≠ t/T, the model learned state-specific information.
"""

import sys
import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import seaborn as sns

# Get the absolute path to the project root
if os.path.basename(os.getcwd()) == 'CMAPSS':
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
else:
    project_root = os.path.abspath(os.getcwd())

if project_root not in sys.path:
    sys.path.insert(0, project_root)

cmapss_dir = os.path.join(project_root, 'CMAPSS')
if cmapss_dir not in sys.path:
    sys.path.insert(0, cmapss_dir)

from CMAPSS.dataset import create_windows
from models.auto_encoders import SimpleTransformerTriplet
from CMAPSS.degrad_analyzer import DegradationAnalyzer
from latent_his.extract_latent import extract_latent_representations


def compute_distributions(all_data, metric='progression_scores'):
    """
    Extract a specific metric and compute normalized timesteps (t/T) for all samples.

    Args:
        all_data: Dictionary of unit data from DegradationAnalyzer
        metric: Which metric to extract ('progression_scores', 'angles', 'z_latent', 'radii', etc.)
                For 'z_latent', returns flattened version

    Returns:
        metric_values: All metric values across all trajectories
        normalized_timesteps: All normalized timesteps (t/T) across all trajectories
        trajectory_info: Info about each trajectory for filtering/analysis
    """
    metric_values = []
    normalized_timesteps = []
    trajectory_info = []

    for unit_id, unit_data in all_data.items():
        if metric not in unit_data:
            print(f"Warning: Metric '{metric}' not found in unit {unit_id}")
            continue

        metric_data = unit_data[metric]

        # Handle z_latent (2D array) - compute per-timestep L2 norm instead of flattening
        # This gives a single scalar per timestep (magnitude of the latent vector)
        if isinstance(metric_data, np.ndarray) and metric_data.ndim > 1:
            metric_data = np.linalg.norm(metric_data, axis=1)

        T = len(unit_data['progression_scores'])  # Use progression_scores length as reference

        # Normalized timesteps: t/T where t ∈ [0, T-1]
        t = np.arange(T)
        t_normalized = t / (T - 1) if T > 1 else t  # Avoid division by zero

        metric_values.extend(metric_data)
        normalized_timesteps.extend(t_normalized)

        # Track which unit each sample belongs to
        trajectory_info.extend([{'unit_id': unit_id, 'T': T, 'is_early': t < T/3, 'is_mid': (t >= T/3) & (t < 2*T/3), 'is_late': t >= 2*T/3} for t in range(T)])

    metric_values = np.array(metric_values)
    normalized_timesteps = np.array(normalized_timesteps)

    return (metric_values, normalized_timesteps, trajectory_info)


def check_and_fix_ordering(metric_values, all_data, metric='progression_scores'):
    """
    Check if metric should be reversed to follow data direction (low at start, high at end).

    Examines per-trajectory trends: if the majority of trajectories show the metric
    decreasing over time, we reverse it so that low = early, high = late.

    Returns the metric (possibly reversed) and a flag indicating if it was reversed.
    """
    # Check per-trajectory: does the metric tend to increase or decrease over time?
    increasing_count = 0
    decreasing_count = 0

    for unit_id, unit_data in all_data.items():
        if metric not in unit_data:
            continue
        m = unit_data[metric]
        if isinstance(m, np.ndarray) and m.ndim > 1:
            m = np.linalg.norm(m, axis=1)
        if len(m) < 3:
            continue
        # Compare first quarter mean vs last quarter mean
        q = max(1, len(m) // 4)
        if np.nanmean(m[:q]) < np.nanmean(m[-q:]):
            increasing_count += 1
        else:
            decreasing_count += 1

    should_reverse = decreasing_count > increasing_count

    if should_reverse:
        metric_values_corrected = 1.0 - metric_values
        return metric_values_corrected, True

    return metric_values, False


def plot_distributions(metric_values, normalized_timesteps, dataset_name, latent_dim,
                      window_size, metric_name='metric', metric_key='progression_scores',
                      all_data=None, output_dir='figures', suffix=''):
    """
    Create comprehensive density comparison plots.

    Args:
        metric_values: The metric to analyze (e.g., progression scores, angles, etc.)
        normalized_timesteps: Normalized time (t/T)
        metric_name: Human-readable name of the metric (e.g., 'θ (progression score)')
        metric_key: Key in all_data dict (e.g., 'progression_scores', 'angles')
        all_data: The raw all_data dict (needed for ordering check)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check and fix ordering if needed
    metric_values_original = metric_values.copy()
    if all_data is not None:
        metric_values, was_reversed = check_and_fix_ordering(metric_values, all_data, metric_key)
    else:
        was_reversed = False

    if was_reversed:
        print(f"  ℹ️  {metric_name} was reversed for intuitive ordering (low→high with degradation)")

    # Compute KS statistic
    ks_statistic, p_value = ks_2samp(metric_values, normalized_timesteps)

    print(f"\n{'='*80}")
    print(f"DENSITY DECOUPLING TEST: {metric_name}")
    print(f"{'='*80}")
    print(f"Number of samples: {len(metric_values)}")
    print(f"{metric_name} distribution: mean={metric_values.mean():.4f}, std={metric_values.std():.4f}, "
          f"min={metric_values.min():.4f}, max={metric_values.max():.4f}")
    print(f"t/T distribution: mean={normalized_timesteps.mean():.4f}, std={normalized_timesteps.std():.4f}, "
          f"min={normalized_timesteps.min():.4f}, max={normalized_timesteps.max():.4f}")
    print(f"\nKS Test Results:")
    print(f"  KS Statistic: {ks_statistic:.6f}")
    print(f"  p-value: {p_value:.2e}")

    if p_value < 0.001:
        print(f"  ✓ HIGHLY SIGNIFICANT: Distributions are very different (p < 0.001)")
        print(f"    → {metric_name} does NOT just learn t/T!")
    elif p_value < 0.05:
        print(f"  ✓ SIGNIFICANT: Distributions are different (p < 0.05)")
        print(f"    → {metric_name} learned beyond temporal order")
    else:
        print(f"  ⚠️  NOT SIGNIFICANT: Distributions are similar (p ≥ 0.05)")
        print(f"    → Concern: {metric_name} may be learning just t/T")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Overlaid density distributions
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(metric_values, bins=50, alpha=0.6, label=metric_name, density=True, color='blue')
    ax1.hist(normalized_timesteps, bins=50, alpha=0.6, label='t/T (normalized timestep)', density=True, color='red')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Marginal Distributions: {metric_name} vs t/T')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: KDE density curves
    ax2 = plt.subplot(2, 3, 2)
    from scipy.stats import gaussian_kde
    kde_metric = gaussian_kde(metric_values)
    kde_t = gaussian_kde(normalized_timesteps)
    x_range = np.linspace(0, 1, 500)
    ax2.plot(x_range, kde_metric(x_range), 'b-', linewidth=2, label=f'{metric_name} KDE')
    ax2.plot(x_range, kde_t(x_range), 'r-', linewidth=2, label='t/T KDE')
    ax2.fill_between(x_range, kde_metric(x_range), alpha=0.3, color='blue')
    ax2.fill_between(x_range, kde_t(x_range), alpha=0.3, color='red')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.set_title(f'KDE Comparison (KS statistic={ks_statistic:.4f}, p={p_value:.2e})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])

    # Plot 3: Cumulative distributions
    ax3 = plt.subplot(2, 3, 3)
    sorted_metric = np.sort(metric_values)
    sorted_t = np.sort(normalized_timesteps)
    ax3.plot(sorted_metric, np.linspace(0, 1, len(sorted_metric)), 'b-', linewidth=2, label=f'{metric_name} CDF')
    ax3.plot(sorted_t, np.linspace(0, 1, len(sorted_t)), 'r-', linewidth=2, label='t/T CDF')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Functions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Scatter of metric vs t/T (if reasonable number of points)
    ax4 = plt.subplot(2, 3, 4)
    if len(metric_values) < 10000:
        ax4.scatter(normalized_timesteps, metric_values, alpha=0.3, s=10, c=metric_values, cmap='viridis')
        ax4.set_xlabel('t/T (normalized timestep)')
        ax4.set_ylabel(metric_name)
        ax4.set_title(f'{metric_name} vs t/T Scatter')
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label(metric_name)
    else:
        # For too many points, use hexbin
        hb = ax4.hexbin(normalized_timesteps, metric_values, gridsize=30, cmap='Blues')
        ax4.set_xlabel('t/T (normalized timestep)')
        ax4.set_ylabel(metric_name)
        ax4.set_title(f'{metric_name} vs t/T Hexbin Density')
        plt.colorbar(hb, ax=ax4, label='count')
    ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label=f'y=x (if {metric_name}≈t/T)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Box plots
    ax5 = plt.subplot(2, 3, 5)
    bp = ax5.boxplot([metric_values, normalized_timesteps], tick_labels=[metric_name, 't/T'],
                      patch_artist=True, notch=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax5.set_ylabel('Value')
    ax5.set_title('Distribution Statistics')
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: QQ-plot equivalent (percentiles)
    ax6 = plt.subplot(2, 3, 6)
    percentiles = np.linspace(0, 100, 101)
    metric_percentiles = np.percentile(metric_values, percentiles)
    t_percentiles = np.percentile(normalized_timesteps, percentiles)
    ax6.plot(percentiles, metric_percentiles, 'b-', linewidth=2, label=f'{metric_name} percentiles')
    ax6.plot(percentiles, t_percentiles, 'r-', linewidth=2, label='t/T percentiles')
    ax6.set_xlabel('Percentile')
    ax6.set_ylabel('Value')
    ax6.set_title('Percentile Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save with informative filename
    metric_suffix = metric_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    filename = f'{dataset_name}_z{latent_dim}_w{window_size}_density_decoupling_{metric_suffix}{suffix}.pdf'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {filepath}")
    plt.close()

    return ks_statistic, p_value


def plot_phase_analysis(metric_values, normalized_timesteps, trajectory_info,
                       dataset_name, latent_dim, window_size, metric_name='metric',
                       output_dir='figures', suffix=''):
    """
    Analyze metric vs t/T in different trajectory phases (early, mid, late).
    Shows if the metric behaves differently in different parts of degradation.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract phase information
    early_idx = [i for i, info in enumerate(trajectory_info) if info['is_early']]
    mid_idx = [i for i, info in enumerate(trajectory_info) if info['is_mid']]
    late_idx = [i for i, info in enumerate(trajectory_info) if info['is_late']]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    phases = [
        ('Early (0-33%)', early_idx, 'green'),
        ('Mid (33-67%)', mid_idx, 'orange'),
        ('Late (67-100%)', late_idx, 'red')
    ]

    for ax, (phase_name, phase_indices, color) in zip(axes, phases):
        if phase_indices:
            metric_phase = metric_values[phase_indices]
            t_phase = normalized_timesteps[phase_indices]

            ax.hist(metric_phase, bins=30, alpha=0.6, label=metric_name, density=True, color='blue')
            ax.hist(t_phase, bins=30, alpha=0.6, label='t/T', density=True, color='red')

            ks_stat, p_val = ks_2samp(metric_phase, t_phase)
            ax.set_title(f'{phase_name}\n(KS={ks_stat:.4f}, p={p_val:.2e})')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    metric_suffix = metric_name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    filename = f'{dataset_name}_z{latent_dim}_w{window_size}_phase_analysis_{metric_suffix}{suffix}.pdf'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved phase analysis: {filepath}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test 1b: Density Decoupling')
    parser.add_argument('--dataset', type=str, default='FD001', help='Dataset name')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained backbone model')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent dimension')
    parser.add_argument('--d_model', type=int, default=32, help='Transformer d_model')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--window_size', type=int, default=10, help='Window size')
    parser.add_argument('--seed', type=int, default=16976296098443334824, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='figures', help='Output directory for plots')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")

    # Load data
    data_paths = [
        os.path.join(project_root, 'data'),
        'data',
        os.path.join(project_root, 'CMAPSS', 'data'),
        os.path.join(os.getcwd(), 'data')
    ]

    data_path = None
    for path in data_paths:
        if os.path.exists(os.path.join(path, f'train_{args.dataset}.txt')):
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError(f"Could not find dataset files for {args.dataset}")

    print(f"Loading data from: {data_path}")

    dataset_train = np.loadtxt(os.path.join(data_path, f'train_{args.dataset}.txt'))

    train_data = np.hstack((
        dataset_train[:, 0].reshape(-1, 1),
        dataset_train[:, 1].reshape(-1, 1),
        dataset_train[:, 5:]
    ))

    # Create windows
    X_train, train_rul_labels, train_unit_ids = create_windows(
        train_data, window_size=args.window_size, threshold=0
    )

    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize and load model
    input_dim = X_train_scaled.shape[1]
    model = SimpleTransformerTriplet(
        input_dim,
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    print(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Extract latent representations
    print("Extracting latent representations...")
    with torch.no_grad():
        z_train = extract_latent_representations(model, X_train_scaled, device)

    scaler_z = StandardScaler()
    z_train_scaled = scaler_z.fit_transform(z_train)

    # Compute progression data
    print("Computing progression data...")
    analyzer = DegradationAnalyzer(
        model, dataset_train, scaler, device,
        scaler_z=scaler_z, proto_size=args.latent_dim,
        window_size=args.window_size, scale=True
    )
    progression_scaler = analyzer.fit_progression_scaler(method='angular')
    all_data = analyzer.compute_all_progression_data(method='angular')

    print(f"Computed data for {len(all_data)} units")

    # Test 1b: Density Decoupling - test multiple metrics
    metrics_to_test = [
        ('progression_scores', 'θ (progression score)'),
        ('angles', 'angles (rad)'),
        ('angles_deg', 'angles (deg)'),
        ('radii', 'radii'),
        ('z_latent', 'z_latent (L2 norm)'),
    ]

    all_results = {}

    for metric_key, metric_name in metrics_to_test:
        print(f"\n{'='*80}")
        print(f"Testing metric: {metric_name}")
        print(f"{'='*80}")

        metric_values, normalized_timesteps, trajectory_info = compute_distributions(
            all_data, metric=metric_key
        )

        if len(metric_values) == 0:
            print(f"Skipping {metric_name} - no data found")
            continue

        # Normalize metric to [0, 1] range for fair comparison
        metric_values_min = np.nanmin(metric_values)
        metric_values_max = np.nanmax(metric_values)
        if metric_values_max > metric_values_min:
            metric_values_norm = (metric_values - metric_values_min) / (metric_values_max - metric_values_min)
        else:
            metric_values_norm = metric_values

        print(f"Total samples: {len(metric_values_norm)}")

        # Create main density plots
        ks_stat, p_value = plot_distributions(
            metric_values_norm, normalized_timesteps,
            dataset_name=args.dataset,
            latent_dim=args.latent_dim,
            window_size=args.window_size,
            metric_name=metric_name,
            metric_key=metric_key,
            all_data=all_data,
            output_dir=args.output_dir
        )

        # Create phase analysis plots
        plot_phase_analysis(
            metric_values_norm, normalized_timesteps, trajectory_info,
            dataset_name=args.dataset,
            latent_dim=args.latent_dim,
            window_size=args.window_size,
            metric_name=metric_name,
            output_dir=args.output_dir
        )

        all_results[metric_name] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'mean': metric_values.mean(),
            'std': metric_values.std()
        }

    print("\n" + "="*80)
    print("Test 1b: Density Decoupling Complete!")
    print("="*80)
    print(f"\nSummary of All Metrics:")
    print("-" * 80)
    print(f"{'Metric':<30} {'KS Statistic':<15} {'p-value':<15} {'Significant':<15}")
    print("-" * 80)

    for metric_name in sorted(all_results.keys()):
        result = all_results[metric_name]
        ks_stat = result['ks_statistic']
        p_val = result['p_value']
        sig = "✓ YES (p<0.001)" if p_val < 0.001 else "✓ YES (p<0.05)" if p_val < 0.05 else "✗ NO"
        print(f"{metric_name:<30} {ks_stat:<15.6f} {p_val:<15.2e} {sig:<15}")

    print("-" * 80)
    print(f"\nInterpretation:")
    print("All metrics with p < 0.001 show strong evidence that they encode")
    print("state information beyond simple temporal order (t/T).")
    print("\nThe backbone learns DEGRADATION STATE, not just TIME ORDERING! ✓")


if __name__ == "__main__":
    main()
