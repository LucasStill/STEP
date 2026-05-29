#!/usr/bin/env python3
"""
Test 1c: Fixed-Timestep Variable-Health Comparison

The cleanest proof that θ tracks STATE, not TIME:
- Find pairs of observations from different engines at the SAME normalized time t/T
- But the engines have DIFFERENT health states (different RUL)
- If θ ≈ t/T, the observations should have similar θ values
- If θ tracks state, the observations should have different θ values

CMAPSS is perfect for this: engines die at different times T, so at t/T=0.5,
an engine that dies at T=100 and one that dies at T=200 are at the same relative
time but with different actual degradation states.
"""

import sys
import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd

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


def find_same_time_different_health_pairs(all_data, quantile_bins=10, max_pairs_per_bucket=500):
    """
    Find pairs of observations at the same normalized time t/T but with different health.

    For each normalized time bucket (e.g., t/T ∈ [0.4, 0.5]), collect observations
    from different units. If they have different RUL, they represent different health
    states at the same relative time.

    To keep the computation tractable, we subsample within each bucket:
    - Pick one representative sample per unit per bucket (the one closest to the bucket center)
    - Cap the number of pairs per bucket

    Returns:
        comparison_pairs: List of {t_norm, theta_A, theta_B, rul_A, rul_B, unit_A, unit_B}
    """
    # Organize samples by normalized time bucket
    # Keep only ONE representative per unit per bucket (closest to bucket center)
    bin_edges = np.linspace(0, 1, quantile_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    time_buckets = {i: {} for i in range(quantile_bins)}  # bucket -> {unit_id: best_sample}

    for unit_id, unit_data in all_data.items():
        progression_scores = unit_data['progression_scores']
        traj_rul = unit_data['traj_rul']
        T = len(progression_scores)

        t = np.arange(T)
        t_normalized = t / (T - 1) if T > 1 else t

        # Assign to buckets
        bucket_indices = np.digitize(t_normalized, bin_edges) - 1
        bucket_indices = np.clip(bucket_indices, 0, quantile_bins - 1)

        for idx, (theta, rul, t_norm, bucket_idx) in enumerate(
            zip(progression_scores, traj_rul, t_normalized, bucket_indices)
        ):
            sample = {
                'unit_id': unit_id,
                'idx': idx,
                't_norm': t_norm,
                'theta': theta,
                'rul': rul,
                'bucket': bucket_idx
            }
            # Keep the sample closest to the bucket center for this unit
            center = bin_centers[bucket_idx]
            if unit_id not in time_buckets[bucket_idx]:
                time_buckets[bucket_idx][unit_id] = sample
            else:
                existing = time_buckets[bucket_idx][unit_id]
                if abs(t_norm - center) < abs(existing['t_norm'] - center):
                    time_buckets[bucket_idx][unit_id] = sample

    # Find pairs at the same time bucket with different RUL
    comparison_pairs = []
    for bucket_idx, unit_samples in time_buckets.items():
        samples = list(unit_samples.values())
        if len(samples) < 2:
            continue

        bucket_pairs = []
        # For each pair of different units in this bucket
        for i, sample_a in enumerate(samples):
            for sample_b in samples[i+1:]:
                # Check if they have meaningfully different health
                rul_diff = abs(sample_a['rul'] - sample_b['rul'])
                if rul_diff > 10:  # At least 10 cycles difference
                    bucket_pairs.append({
                        't_norm': (sample_a['t_norm'] + sample_b['t_norm']) / 2,
                        'bucket': bucket_idx,
                        'theta_A': sample_a['theta'],
                        'theta_B': sample_b['theta'],
                        'theta_diff': abs(sample_a['theta'] - sample_b['theta']),
                        'rul_A': sample_a['rul'],
                        'rul_B': sample_b['rul'],
                        'rul_diff': rul_diff,
                        'unit_A': sample_a['unit_id'],
                        'unit_B': sample_b['unit_id'],
                    })

        # Subsample if too many pairs in this bucket
        if len(bucket_pairs) > max_pairs_per_bucket:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(bucket_pairs), max_pairs_per_bucket, replace=False)
            bucket_pairs = [bucket_pairs[i] for i in indices]

        comparison_pairs.extend(bucket_pairs)

    return comparison_pairs


def plot_same_time_different_health(comparison_pairs, dataset_name, latent_dim, window_size,
                                   output_dir='figures'):
    """
    Visualize whether θ differs for observations at same time but different health.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not comparison_pairs:
        print("No comparison pairs found")
        return

    df = pd.DataFrame(comparison_pairs)

    print(f"\n{'='*80}")
    print(f"SAME-TIME DIFFERENT-HEALTH TEST: {dataset_name}")
    print(f"{'='*80}")
    print(f"Found {len(df)} comparison pairs at same relative time with different health")
    print(f"\nRUL difference range: {df['rul_diff'].min():.1f} - {df['rul_diff'].max():.1f} cycles")
    print(f"θ difference range: {df['theta_diff'].min():.4f} - {df['theta_diff'].max():.4f}")

    # Compute correlation between health difference and θ difference
    corr_ruldiff_thetadiff, p_corr = spearmanr(df['rul_diff'], df['theta_diff'])
    print(f"\nSpearman correlation (RUL diff vs θ diff): {corr_ruldiff_thetadiff:.4f} (p={p_corr:.2e})")

    if p_corr < 0.05 and corr_ruldiff_thetadiff > 0.3:
        print("✓ STRONG: Different health → Different θ values")
        print("  θ tracks HEALTH STATE, not just temporal order!")
    elif p_corr < 0.05:
        print("✓ SIGNIFICANT: Health difference correlates with θ difference")
        print("  θ encodes state-specific information beyond temporal order")
    else:
        print("️  No significant correlation between health and θ difference")
        print("  (θ may be similar regardless of health at the same relative time)")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Scatter of RUL difference vs θ difference
    # Subsample for plotting if too many points
    ax = axes[0, 0]
    plot_df = df.sample(min(2000, len(df)), random_state=42) if len(df) > 2000 else df
    scatter = ax.scatter(plot_df['rul_diff'], plot_df['theta_diff'], alpha=0.4, c=plot_df['t_norm'], cmap='viridis', s=30)
    ax.set_xlabel('RUL Difference (cycles)')
    ax.set_ylabel('θ Difference')
    ax.set_title(f'Health State Difference vs θ Difference\n(Spearman ρ={corr_ruldiff_thetadiff:.4f}, p={p_corr:.2e})')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized Time (t/T)')

    # Add trend line (computed on full data, plotted)
    z = np.polyfit(df['rul_diff'], df['theta_diff'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['rul_diff'].min(), df['rul_diff'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=2, label='Trend')
    ax.legend(loc='upper left')

    # Plot 2: Distribution of θ differences across time buckets
    ax = axes[0, 1]
    buckets = sorted(df['bucket'].unique())
    bucket_theta_diffs = [df[df['bucket'] == b]['theta_diff'].values for b in buckets]
    bp = ax.boxplot(bucket_theta_diffs, tick_labels=[f't/T={b/len(buckets):.1f}' for b in buckets])
    ax.set_ylabel('θ Difference')
    ax.set_xlabel('Normalized Time Bucket')
    ax.set_title('θ Difference Across Different Times')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Scatter matrix showing examples
    ax = axes[1, 0]
    # Sample pairs for visualization
    sample_pairs = df.sample(min(50, len(df))).reset_index(drop=True)
    colors = plt.cm.viridis(sample_pairs['t_norm'].values)
    for loop_idx, (idx, row) in enumerate(sample_pairs.iterrows()):
        ax.plot([row['theta_A'], row['theta_B']], [0, 1], 'o-', color=colors[loop_idx], alpha=0.5)
    ax.set_xlim([df['theta_A'].min() - 0.05, df['theta_A'].max() + 0.05])
    ax.set_ylabel('Unit Pair')
    ax.set_xlabel('θ values (comparing pairs)')
    ax.set_title('Sample Pairs: Showing θ Differences Within Pairs')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Unit A', 'Unit B'])

    # Plot 4: Statistics summary
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
Test 1c: Same-Time Different-Health Analysis

Dataset: {dataset_name}
Latent Dim: {latent_dim}
Window Size: {window_size}

Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Comparison Pairs: {len(df)}

RUL Difference:
  Mean: {df['rul_diff'].mean():.1f} cycles
  Median: {df['rul_diff'].median():.1f} cycles
  Max: {df['rul_diff'].max():.1f} cycles

θ Difference:
  Mean: {df['theta_diff'].mean():.4f}
  Median: {df['theta_diff'].median():.4f}
  Max: {df['theta_diff'].max():.4f}

Correlation:
  Spearman ρ = {corr_ruldiff_thetadiff:.4f}
  p-value = {p_corr:.2e}

Interpretation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If ρ >> 0 and p < 0.05:
  → θ TRACKS STATE (different health
     leads to different θ even at same t/T)

If ρ ≈ 0 or p > 0.05:
  → θ may be learning just t/T
     (health difference doesn't affect θ)
"""
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontfamily='monospace',
           verticalalignment='top', fontsize=9, bbox=dict(boxstyle='round',
           facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    filename = f'{dataset_name}_z{latent_dim}_w{window_size}_same_time_different_health.pdf'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved plot: {filepath}")
    plt.close()

    # Save detailed CSV
    csv_filename = f'{dataset_name}_z{latent_dim}_w{window_size}_comparison_pairs.csv'
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved detailed results: {csv_path}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test 1c: Same-Time Different-Health')
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

    # Test 1c: Find and analyze pairs
    comparison_pairs = find_same_time_different_health_pairs(all_data, quantile_bins=10)

    if comparison_pairs:
        df = plot_same_time_different_health(
            comparison_pairs,
            dataset_name=args.dataset,
            latent_dim=args.latent_dim,
            window_size=args.window_size,
            output_dir=args.output_dir
        )

        print("\n" + "="*80)
        print("Test 1c: Same-Time Different-Health Complete!")
        print("="*80)
    else:
        print("Could not find comparison pairs in the data")


if __name__ == "__main__":
    main()
