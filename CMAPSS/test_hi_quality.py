#!/usr/bin/env python3
"""
Health Indicator Quality Assessment — Formal characterization of θ as a state representation.

This script answers: "What IS θ?" by showing it is a valid health indicator that
captures degradation state beyond temporal order.

Three layers of evidence:
  1. Standard PHM metrics (Mon/Tre/Pro) + Spearman correlation with RUL
  2. Partial correlation with RUL controlling for t/T — proves state ≠ time
  3. Conditional predictive gain using the actual downstream transformer

References:
  - Coble & Hines (2009): Monotonicity, Trendability, Prognosability
  - Javed et al. (2015): HI quality criteria for PHM
"""

import sys
import os
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Path setup
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


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: Standard PHM Health Indicator Metrics (Coble & Hines 2009)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_monotonicity(values):
    """
    Monotonicity: fraction of consecutive pairs that follow a consistent trend.
    Mon = |#(positive diffs) - #(negative diffs)| / #(diffs)
    Perfect monotonic signal → 1.0. Random noise → ~0.0.
    """
    diffs = np.diff(values)
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    n_total = len(diffs)
    if n_total == 0:
        return 0.0
    return abs(n_pos - n_neg) / n_total


def compute_trendability(all_trajectories, all_rul_trajectories):
    """
    Trendability: consistency of the HI's correlation with RUL across units.

    Tre = mean of |corr(HI_k, RUL_k)| across all units k.

    Note: correlation with RUL (not with time) is the correct formulation —
    it measures whether the HI tracks the actual degradation state consistently.
    """
    correlations = []
    for traj, rul in zip(all_trajectories, all_rul_trajectories):
        if len(traj) < 3 or len(rul) < 3:
            continue
        corr, _ = spearmanr(traj, rul)
        if not np.isnan(corr):
            correlations.append(abs(corr))

    if not correlations:
        return 0.0
    return np.mean(correlations)


def compute_prognosability(all_trajectories):
    """
    Prognosability: how tightly do HI values cluster at failure?
    Pro = exp(-std(HI_at_failure) / mean_range(HI))
    Low variance at failure → high prognosability.
    """
    failure_values = []
    ranges = []
    for traj in all_trajectories:
        if len(traj) < 2:
            continue
        failure_values.append(traj[-1])
        ranges.append(np.ptp(traj))

    if not failure_values or np.mean(ranges) == 0:
        return 0.0

    return np.exp(-np.std(failure_values) / np.mean(ranges))


def compute_initial_distinguishability(all_trajectories):
    """
    Initial Distinguishability: how tightly do HI values cluster at the START?
    ID = exp(-std(HI_at_start) / mean_range(HI))
    Low variance at healthy state → high initial distinguishability.
    """
    start_values = []
    ranges = []
    for traj in all_trajectories:
        if len(traj) < 2:
            continue
        start_values.append(traj[0])
        ranges.append(np.ptp(traj))

    if not start_values or np.mean(ranges) == 0:
        return 0.0

    return np.exp(-np.std(start_values) / np.mean(ranges))


def compute_spearman_with_rul(all_trajectories, all_rul_trajectories):
    """
    Mean per-unit |Spearman correlation| between HI and RUL.
    This is the single most important scalar: does the HI track health?
    """
    correlations = []
    for traj, rul in zip(all_trajectories, all_rul_trajectories):
        if len(traj) < 3:
            continue
        corr, _ = spearmanr(traj, rul)
        if not np.isnan(corr):
            correlations.append(abs(corr))
    return np.mean(correlations) if correlations else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: Partial Correlation — The formal proof that θ ≠ t/T
# ═══════════════════════════════════════════════════════════════════════════════

def partial_correlation(x, y, z):
    """
    Compute partial correlation between x and y, controlling for z.
    pcorr(x, y | z) = corr(residual_x, residual_y)
    """
    z = z.reshape(-1, 1) if z.ndim == 1 else z

    reg_x = LinearRegression().fit(z, x)
    residual_x = x - reg_x.predict(z)

    reg_y = LinearRegression().fit(z, y)
    residual_y = y - reg_y.predict(z)

    # Check for near-zero variance in residuals
    if np.std(residual_x) < 1e-10 or np.std(residual_y) < 1e-10:
        return 0.0, 1.0

    corr, p_value = pearsonr(residual_x, residual_y)
    return corr, p_value


def compute_pooled_partial_correlation(all_data):
    """
    Compute partial correlation on ALL data pooled across units.

    This avoids per-unit collinearity issues (short trajectories where θ ≈ linear)
    and tests the population-level claim: does θ carry information about RUL
    that t/T cannot explain, across the entire dataset?

    Also computes per-unit partial correlations for the distribution plot,
    filtering out degenerate cases.
    """
    # Pool all data
    all_theta = []
    all_rul = []
    all_t_norm = []

    for unit_id, unit_data in all_data.items():
        theta = unit_data['progression_scores']
        rul = unit_data['traj_rul']
        T = len(theta)
        if T < 3:
            continue
        t_norm = np.arange(T, dtype=np.float64) / (T - 1)
        all_theta.extend(theta)
        all_rul.extend(rul)
        all_t_norm.extend(t_norm)

    all_theta = np.array(all_theta, dtype=np.float64)
    all_rul = np.array(all_rul, dtype=np.float64)
    all_t_norm = np.array(all_t_norm, dtype=np.float64)

    # Pooled partial correlation
    pooled_pcorr, pooled_p = partial_correlation(all_theta, all_rul, all_t_norm)

    # Per-unit partial correlations (for distribution plot)
    per_unit_results = []
    for unit_id, unit_data in all_data.items():
        theta = unit_data['progression_scores']
        rul = unit_data['traj_rul']
        T = len(theta)
        if T < 10:  # Need enough points for meaningful partial correlation
            continue

        t_norm = np.arange(T, dtype=np.float64) / (T - 1)

        corr_theta_rul, _ = spearmanr(theta, rul)
        corr_t_rul, _ = spearmanr(t_norm, rul)

        pcorr, p_pcorr = partial_correlation(
            np.array(theta, dtype=np.float64),
            np.array(rul, dtype=np.float64),
            np.array(t_norm, dtype=np.float64)
        )

        if not np.isnan(pcorr):
            per_unit_results.append({
                'unit_id': unit_id,
                'T': T,
                'corr_theta_rul': corr_theta_rul,
                'corr_t_rul': corr_t_rul,
                'pcorr_theta_rul_given_t': pcorr,
                'p_pcorr': p_pcorr,
            })

    return pooled_pcorr, pooled_p, per_unit_results


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_metric_trajectories(all_data, metric_key):
    """Extract per-unit trajectories. For z_latent, computes L2 norm per timestep."""
    trajectories = []
    rul_trajectories = []
    unit_ids = []

    for unit_id, unit_data in all_data.items():
        if metric_key not in unit_data:
            continue
        vals = unit_data[metric_key]
        if isinstance(vals, np.ndarray) and vals.ndim > 1:
            vals = np.linalg.norm(vals, axis=1)
        trajectories.append(np.array(vals, dtype=np.float64))
        rul_trajectories.append(np.array(unit_data['traj_rul'], dtype=np.float64))
        unit_ids.append(unit_id)

    return trajectories, rul_trajectories, unit_ids


def normalize_trajectories(trajectories):
    """Normalize to [0, 1] globally. Ensures increasing direction with degradation."""
    all_vals = np.concatenate(trajectories)
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    if vmax == vmin:
        return [np.zeros_like(t) for t in trajectories]

    normalized = [(t - vmin) / (vmax - vmin) for t in trajectories]

    # Check majority direction
    increasing = sum(1 for t in normalized if len(t) >= 3 and
                     np.nanmean(t[:max(1, len(t)//4)]) < np.nanmean(t[-max(1, len(t)//4):]))
    decreasing = sum(1 for t in normalized if len(t) >= 3 and
                     np.nanmean(t[:max(1, len(t)//4)]) >= np.nanmean(t[-max(1, len(t)//4):]))

    if decreasing > increasing:
        normalized = [1.0 - t for t in normalized]

    return normalized


def inject_t_normalized(all_data):
    """Add t_normalized key to each unit's data dict for benchmark_model compatibility."""
    for unit_id, unit_data in all_data.items():
        T = len(unit_data['progression_scores'])
        t_norm = np.arange(T, dtype=np.float64) / (T - 1) if T > 1 else np.zeros(T)
        unit_data['t_normalized'] = t_norm
    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_hi_quality_figure(metrics_table, pooled_pcorr, pooled_p, per_unit_results,
                             predictive_results, all_data, dataset_name, latent_dim,
                             window_size, output_dir):
    """Create all HI quality figures."""

    os.makedirs(output_dir, exist_ok=True)
    metric_names = list(metrics_table.keys())
    n_metrics = len(metric_names)

    # ── Figure 1: Mon/Tre/Pro + Spearman bar chart ──
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    properties = ['Monotonicity', 'Trendability', 'Prognosability',
                  'Init. Distinguish.', '|ρ(HI, RUL)|']
    prop_keys = ['monotonicity', 'trendability', 'prognosability',
                 'initial_distinguishability', 'spearman_rul']

    colors_map = {
        'θ (progression)': '#2196F3',
        'angles (rad)': '#4CAF50',
        'radii': '#FF9800',
        '‖z‖ (latent norm)': '#9C27B0',
        't/T (baseline)': '#F44336',
    }

    for i, (prop_name, prop_key) in enumerate(zip(properties, prop_keys)):
        ax = axes[i]
        vals = [metrics_table[m][prop_key] for m in metric_names]
        bar_colors = [colors_map.get(m, '#666666') for m in metric_names]
        bars = ax.bar(range(n_metrics), vals, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels([m.split('(')[0].strip() for m in metric_names],
                           rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(prop_name, fontsize=9)
        ax.set_title(prop_name, fontweight='bold', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle(f'Health Indicator Quality — {dataset_name} (z={latent_dim}, w={window_size})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'{dataset_name}_z{latent_dim}_w{window_size}_hi_quality_bars.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filepath}")

    # ── Figure 2: Partial correlation ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    pcorrs = [r['pcorr_theta_rul_given_t'] for r in per_unit_results]
    raw_corrs = [r['corr_theta_rul'] for r in per_unit_results]
    t_corrs = [r['corr_t_rul'] for r in per_unit_results]

    # Histogram of per-unit partial correlations
    ax = axes[0]
    ax.hist(pcorrs, bins=20, color='#2196F3', alpha=0.7, edgecolor='black',
            label=f'Per-unit pcorr')
    mean_pcorr = np.mean(pcorrs) if pcorrs else 0
    ax.axvline(x=mean_pcorr, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {mean_pcorr:.3f}')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    # Mark the pooled result
    ax.axvline(x=pooled_pcorr, color='darkgreen', linestyle='-', linewidth=2.5,
               label=f'Pooled = {pooled_pcorr:.3f} (p={pooled_p:.1e})')
    ax.set_xlabel('Partial Correlation')
    ax.set_ylabel('Number of Units')
    ax.set_title('pcorr(θ, RUL | t/T)', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Compare: θ-RUL vs t/T-RUL correlation per unit
    ax = axes[1]
    ax.scatter(t_corrs, raw_corrs, alpha=0.6, c='#2196F3', edgecolors='black', linewidth=0.5, s=40)
    lims = [-1.05, 1.05]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='y = x (θ = t/T)')
    ax.set_xlabel('corr(t/T, RUL)')
    ax.set_ylabel('corr(θ, RUL)')
    ax.set_title('Per-Unit: θ vs t/T\ncorrelation with RUL', fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    # Significance of per-unit partial correlations
    ax = axes[2]
    p_values = [r['p_pcorr'] for r in per_unit_results]
    significant = sum(1 for p in p_values if p < 0.05)
    total = len(p_values)
    log_p = [-np.log10(max(p, 1e-300)) for p in p_values]
    ax.bar(range(total), sorted(log_p, reverse=True), color='#2196F3', alpha=0.7, edgecolor='none')
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=1.5,
               label=f'p=0.05 ({significant}/{total} sig.)')
    ax.set_xlabel('Unit (sorted by significance)')
    ax.set_ylabel('-log₁₀(p-value)')
    ax.set_title('Significance of\npcorr(θ, RUL | t/T)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Partial Correlation Analysis — {dataset_name} (z={latent_dim}, w={window_size})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'{dataset_name}_z{latent_dim}_w{window_size}_partial_correlation.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filepath}")

    # ── Figure 3: Predictive gain (downstream transformer) ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    models = list(predictive_results.keys())
    bar_colors_pred = ['#F44336', '#2196F3', '#4CAF50']

    ax = axes[0]
    rmses = [predictive_results[m]['rmse'] for m in models]
    bars = ax.bar(models, rmses, color=bar_colors_pred[:len(models)], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('RUL Prediction RMSE\n(Downstream Transformer)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax = axes[1]
    r2s = [predictive_results[m]['r2'] for m in models]
    bars = ax.bar(models, r2s, color=bar_colors_pred[:len(models)], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('R² (higher is better)')
    ax.set_title('RUL Prediction R²\n(Downstream Transformer)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle(f'Conditional Predictive Gain — {dataset_name} (z={latent_dim}, w={window_size})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'{dataset_name}_z{latent_dim}_w{window_size}_predictive_gain.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filepath}")

    # ── Figure 4: Example trajectories — θ vs t/T vs RUL ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    unit_ids = sorted(all_data.keys())

    lengths = [(uid, len(all_data[uid]['progression_scores'])) for uid in unit_ids]
    lengths.sort(key=lambda x: x[1])
    indices = np.linspace(0, len(lengths) - 1, 6, dtype=int)
    sample_units = [lengths[i][0] for i in indices]

    for ax_idx, unit_id in enumerate(sample_units):
        ax = axes[ax_idx // 3, ax_idx % 3]
        unit_data = all_data[unit_id]

        theta = unit_data['progression_scores']
        rul = unit_data['traj_rul']
        T = len(theta)
        t_norm = np.arange(T, dtype=np.float64) / (T - 1)

        rul_norm = (rul - rul.min()) / (rul.max() - rul.min()) if rul.max() > rul.min() else rul

        ax.plot(range(T), t_norm, 'r-', alpha=0.7, linewidth=1.5, label='t/T')
        ax.plot(range(T), theta, 'b-', linewidth=2, label='θ')
        ax.plot(range(T), 1 - rul_norm, 'k--', alpha=0.5, linewidth=1, label='1−RUL (norm)')
        ax.set_title(f'Unit {unit_id} (T={T})', fontsize=9)
        ax.set_xlabel('Timestep', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    fig.suptitle(f'θ vs t/T vs RUL — {dataset_name} (z={latent_dim}, w={window_size})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'{dataset_name}_z{latent_dim}_w{window_size}_trajectory_comparison.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Health Indicator Quality Assessment')
    parser.add_argument('--dataset', type=str, default='FD001')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=16976296098443334824)
    parser.add_argument('--output_dir', type=str, default='figures')
    # Downstream transformer parameters (match your benchmark_model defaults)
    parser.add_argument('--transformer_epochs', type=int, default=20)
    parser.add_argument('--transformer_lr', type=float, default=0.001)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device} | Dataset: {args.dataset} | z={args.latent_dim} | w={args.window_size}")

    # ── Load data ──
    data_paths = [
        os.path.join(project_root, 'data'), 'data',
        os.path.join(project_root, 'CMAPSS', 'data'),
    ]
    data_path = None
    for path in data_paths:
        if os.path.exists(os.path.join(path, f'train_{args.dataset}.txt')):
            data_path = path
            break
    if data_path is None:
        raise FileNotFoundError(f"Could not find dataset for {args.dataset}")

    dataset_train = np.loadtxt(os.path.join(data_path, f'train_{args.dataset}.txt'))
    dataset_test = np.loadtxt(os.path.join(data_path, f'test_{args.dataset}.txt'))
    dataset_test_RUL = np.loadtxt(os.path.join(data_path, f'RUL_{args.dataset}.txt'))

    train_data = np.hstack((
        dataset_train[:, 0].reshape(-1, 1),
        dataset_train[:, 1].reshape(-1, 1),
        dataset_train[:, 5:]
    ))

    test_data = np.hstack((
        dataset_test[:, 0].reshape(-1, 1),
        dataset_test[:, 1].reshape(-1, 1),
        dataset_test[:, 5:]
    ))

    X_train, train_rul_labels, train_unit_ids = create_windows(
        train_data, window_size=args.window_size, threshold=0
    )

    X_test, test_rul_labels, test_unit_ids = create_windows(
        test_data, window_size=args.window_size, threshold=0,
        dataset_test_RUL=dataset_test_RUL
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Load model ──
    input_dim = X_train_scaled.shape[1]
    model = SimpleTransformerTriplet(
        input_dim, latent_dim=args.latent_dim, d_model=args.d_model,
        nhead=args.nhead, num_layers=args.num_layers, dropout=args.dropout
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        z_train = extract_latent_representations(model, X_train_scaled, device)
    scaler_z = StandardScaler()
    scaler_z.fit_transform(z_train)

    # ── Compute progression data ──
    print("Computing progression data...")
    analyzer = DegradationAnalyzer(
        model, dataset_train, scaler, device,
        scaler_z=scaler_z, proto_size=args.latent_dim,
        window_size=args.window_size, scale=True
    )
    progression_scaler = analyzer.fit_progression_scaler(method='angular')
    all_data = analyzer.compute_all_progression_data(method='angular')

    analyzer_test = DegradationAnalyzer(
        model, dataset_test, scaler, device,
        scaler_z=scaler_z, proto_size=args.latent_dim,
        window_size=args.window_size,
        progression_scaler=progression_scaler, scale=True
    )
    all_data_test = analyzer_test.compute_all_progression_data(method='angular')

    print(f"Computed data for {len(all_data)} train units, {len(all_data_test)} test units\n")

    # Inject t/T into all_data for benchmark_model compatibility
    inject_t_normalized(all_data)
    inject_t_normalized(all_data_test)

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1: Mon/Tre/Pro + Spearman(HI, RUL)
    # ══════════════════════════════════════════════════════════════════════════

    metrics_to_test = [
        ('progression_scores', 'θ (progression)'),
        ('angles', 'angles (rad)'),
        ('radii', 'radii'),
        ('z_latent', '‖z‖ (latent norm)'),
    ]

    metrics_table = {}

    # t/T baseline
    t_trajectories = []
    t_rul_trajectories = []
    for unit_id, unit_data in all_data.items():
        T = len(unit_data['progression_scores'])
        if T < 2:
            continue
        t_norm = np.arange(T, dtype=np.float64) / (T - 1)
        t_trajectories.append(t_norm)
        t_rul_trajectories.append(np.array(unit_data['traj_rul'], dtype=np.float64))

    metrics_table['t/T (baseline)'] = {
        'monotonicity': np.mean([compute_monotonicity(t) for t in t_trajectories]),
        'trendability': compute_trendability(t_trajectories, t_rul_trajectories),
        'prognosability': compute_prognosability(t_trajectories),
        'initial_distinguishability': compute_initial_distinguishability(t_trajectories),
        'spearman_rul': compute_spearman_with_rul(t_trajectories, t_rul_trajectories),
    }

    for metric_key, metric_name in metrics_to_test:
        trajectories, rul_trajectories, _ = extract_metric_trajectories(all_data, metric_key)
        trajectories = normalize_trajectories(trajectories)

        mon_per_unit = [compute_monotonicity(t) for t in trajectories if len(t) >= 3]

        metrics_table[metric_name] = {
            'monotonicity': np.mean(mon_per_unit) if mon_per_unit else 0.0,
            'trendability': compute_trendability(trajectories, rul_trajectories),
            'prognosability': compute_prognosability(trajectories),
            'initial_distinguishability': compute_initial_distinguishability(trajectories),
            'spearman_rul': compute_spearman_with_rul(trajectories, rul_trajectories),
        }

    print("=" * 85)
    print("LAYER 1: PHM Health Indicator Quality Metrics")
    print("=" * 85)
    print(f"\n{'Metric':<22s} {'Mon':>8s} {'Tre':>8s} {'Pro':>8s} {'ID':>8s} {'|ρ(RUL)|':>8s}")
    print("-" * 62)
    for name, vals in metrics_table.items():
        marker = "  ◄ trivial by construction" if "baseline" in name else ""
        print(f"{name:<22s} {vals['monotonicity']:8.3f} {vals['trendability']:8.3f} "
              f"{vals['prognosability']:8.3f} {vals['initial_distinguishability']:8.3f} "
              f"{vals['spearman_rul']:8.3f}{marker}")

    print(f"\n  Note: t/T has perfect Mon/Tre/Pro by construction (always [0→1] linearly).")
    print(f"  The key discriminator is |ρ(HI, RUL)| — correlation with actual health state.")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 2: Partial correlation (pooled + per-unit)
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 85}")
    print("LAYER 2: Partial Correlation — pcorr(θ, RUL | t/T)")
    print("=" * 85)

    pooled_pcorr, pooled_p, per_unit_results = compute_pooled_partial_correlation(all_data)

    print(f"\n  POOLED (all {sum(len(d['progression_scores']) for d in all_data.values())} samples):")
    print(f"    pcorr(θ, RUL | t/T) = {pooled_pcorr:.4f}  (p = {pooled_p:.2e})")

    if per_unit_results:
        pcorrs = [r['pcorr_theta_rul_given_t'] for r in per_unit_results]
        p_vals = [r['p_pcorr'] for r in per_unit_results]
        n_significant = sum(1 for p in p_vals if p < 0.05)

        print(f"\n  PER-UNIT (units with T ≥ 10: {len(per_unit_results)}):")
        print(f"    Mean:   {np.mean(pcorrs):.4f}")
        print(f"    Median: {np.median(pcorrs):.4f}")
        print(f"    Significant (p < 0.05): {n_significant}/{len(per_unit_results)} "
              f"({100 * n_significant / len(per_unit_results):.1f}%)")

    if pooled_p < 0.001 and abs(pooled_pcorr) > 0.05:
        print(f"\n  ✓ SIGNIFICANT: After removing all information that t/T can explain,")
        print(f"    θ still predicts RUL (pcorr={pooled_pcorr:.4f}, p={pooled_p:.2e}).")
        print(f"    → θ encodes STATE information beyond temporal order.")
    else:
        print(f"\n  ⚠ Pooled partial correlation is weak or non-significant.")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 3: Predictive gain using YOUR downstream transformer
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 85}")
    print("LAYER 3: Conditional Predictive Gain (Downstream Transformer)")
    print("=" * 85)

    from models.downstream_head import benchmark_model

    feature_configs = [
        ('t/T only', ['t_normalized']),
        ('θ only', ['progression_scores']),
        ('θ + t/T', ['progression_scores', 't_normalized']),
    ]

    predictive_results = {}
    for label, feature_names in feature_configs:
        print(f"\n  Running: {label} → features={feature_names}")
        try:
            _, y_pred, results = benchmark_model(
                all_data, all_data_test, dataset_test_RUL,
                feature_names=feature_names,
                model_type='transformer',
                smoothing_window=5,
                transformer_epochs=args.transformer_epochs,
                transformer_lr=args.transformer_lr,
                dataset_name=f"{args.dataset}_HIquality_{label.replace(' ', '_')}",
                sheet_name='hi_quality_test',
                record_benchmark=False,
            )
            predictive_results[label] = {
                'rmse': results['rmse'],
                'r2': results.get('r2', 0),
                'mae': results.get('mae', 0),
            }
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            predictive_results[label] = {'rmse': float('inf'), 'r2': 0, 'mae': float('inf')}

    print(f"\n  {'Model':<18s} {'RMSE':>10s} {'R²':>10s} {'MAE':>10s}")
    print(f"  {'-' * 50}")
    for label in [l for l, _ in feature_configs]:
        r = predictive_results[label]
        print(f"  {label:<18s} {r['rmse']:10.2f} {r['r2']:10.4f} {r['mae']:10.2f}")

    rmse_t = predictive_results['t/T only']['rmse']
    rmse_theta = predictive_results['θ only']['rmse']
    rmse_both = predictive_results['θ + t/T']['rmse']

    if rmse_theta < rmse_t:
        gain = (rmse_t - rmse_theta) / rmse_t * 100
        print(f"\n  ✓ θ alone outperforms t/T by {gain:.1f}% RMSE reduction")
        print(f"    → θ learned MORE than temporal order")
    else:
        gap = (rmse_theta - rmse_t) / rmse_t * 100
        print(f"\n  ⚠ t/T alone outperforms θ by {gap:.1f}% — but see complementarity below")

    if rmse_both < min(rmse_t, rmse_theta):
        gain_over_t = (rmse_t - rmse_both) / rmse_t * 100
        gain_over_theta = (rmse_theta - rmse_both) / rmse_theta * 100
        print(f"  ✓ θ + t/T outperforms both individually:")
        print(f"    vs t/T alone:  {gain_over_t:.1f}% RMSE reduction")
        print(f"    vs θ alone:    {gain_over_theta:.1f}% RMSE reduction")
        print(f"    → θ and t/T carry COMPLEMENTARY information (θ ≠ t/T)")

    # ══════════════════════════════════════════════════════════════════════════
    # Generate figures
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 85}")
    print("Generating figures...")
    print("=" * 85)

    create_hi_quality_figure(
        metrics_table, pooled_pcorr, pooled_p, per_unit_results,
        predictive_results, all_data, args.dataset, args.latent_dim,
        args.window_size, args.output_dir
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════

    theta_spearman = metrics_table['θ (progression)']['spearman_rul']
    t_spearman = metrics_table['t/T (baseline)']['spearman_rul']

    print(f"\n{'=' * 85}")
    print("SUMMARY")
    print("=" * 85)
    print(f"""
Dataset: {args.dataset}, z={args.latent_dim}, w={args.window_size}

Layer 1 — PHM Quality (Mon/Tre/Pro + RUL correlation):
  θ:   Mon={metrics_table['θ (progression)']['monotonicity']:.3f}  Tre={metrics_table['θ (progression)']['trendability']:.3f}  Pro={metrics_table['θ (progression)']['prognosability']:.3f}  |ρ(RUL)|={theta_spearman:.3f}
  t/T: Mon={metrics_table['t/T (baseline)']['monotonicity']:.3f}  Tre={metrics_table['t/T (baseline)']['trendability']:.3f}  Pro={metrics_table['t/T (baseline)']['prognosability']:.3f}  |ρ(RUL)|={t_spearman:.3f}
  Note: t/T has trivially perfect Mon/Tre/Pro but |ρ(RUL)| reveals actual predictive value.

Layer 2 — Partial Correlation:
  Pooled pcorr(θ, RUL | t/T) = {pooled_pcorr:.4f}  (p = {pooled_p:.2e})
  → After removing time's contribution, θ still predicts health state.

Layer 3 — Downstream Transformer (your actual benchmark model):
  RMSE: t/T={rmse_t:.1f}, θ={rmse_theta:.1f}, θ+t/T={rmse_both:.1f}
""")

    # Final verdict
    evidence_count = 0
    evidence_details = []

    if abs(pooled_pcorr) > 0.05 and pooled_p < 0.05:
        evidence_count += 1
        evidence_details.append("partial correlation significant")

    if rmse_both < min(rmse_t, rmse_theta):
        evidence_count += 1
        evidence_details.append("complementary predictive gain")

    if rmse_theta < rmse_t:
        evidence_count += 1
        evidence_details.append("θ outperforms t/T")

    if theta_spearman > t_spearman:
        evidence_count += 1
        evidence_details.append("higher RUL correlation than t/T")

    if evidence_count >= 2:
        print("CONCLUSION: Strong evidence that θ captures degradation STATE beyond")
        print("temporal order. Evidence: " + "; ".join(evidence_details) + ".")
    elif evidence_count == 1:
        print("CONCLUSION: Partial evidence that θ encodes state information.")
        print("Evidence: " + "; ".join(evidence_details) + ".")
    else:
        print("CONCLUSION: Limited evidence for θ as a state representation.")
        print("Consider retraining with different hyperparameters.")

    print(f"\n{'=' * 85}")


if __name__ == "__main__":
    main()
