#!/usr/bin/env python3
"""
Test script to check if the backbone model has memorized timesteps rather than state representations.

This script loads the trained backbone and benchmark data, but replaces all features with
simple timestep indices. If the downstream model can still predict RUL, it suggests the
backbone may be memorizing timestep patterns rather than learning meaningful state representations.
"""

import sys
import os
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
from models.downstream_head import benchmark_model
from latent_his.extract_latent import extract_latent_representations


def create_timestep_data(all_data_original):
    """
    Replace all features in all_data with simple timestep indices.

    For each unit, we replace all features with an array of timesteps [0, 1, 2, ..., T-1]
    where T is the trajectory length.
    """
    all_data_timesteps = {}

    for unit_id, unit_data in all_data_original.items():
        T = len(unit_data['progression_scores'])  # Use progression_scores length as reference

        # Create timestep array
        timesteps = np.arange(T, dtype=np.float32)

        # Create a new data dictionary with same structure but only timestep values
        all_data_timesteps[unit_id] = {
            'progression_scores': timesteps,
            'angles': timesteps,
            'radii': timesteps,
            'angles_deg': timesteps,
            'z_latent': np.repeat(timesteps.reshape(-1, 1), 16, axis=1),  # Match latent dim
        }

    return all_data_timesteps


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test timestep memorization')
    parser.add_argument('--dataset', type=str, default='FD001', help='Dataset name')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained backbone model')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent dimension')
    parser.add_argument('--d_model', type=int, default=32, help='Transformer d_model')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--window_size', type=int, default=10, help='Window size')
    parser.add_argument('--seed', type=int, default=16976296098443334824, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Model path: {args.model_path}")

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

    # Create windows
    X_train, train_rul_labels, train_unit_ids = create_windows(
        train_data, window_size=args.window_size, threshold=0
    )

    X_test, test_rul_labels, test_unit_ids = create_windows(
        test_data, window_size=args.window_size, threshold=0,
        dataset_test_RUL=dataset_test_RUL
    )

    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    # Load pretrained weights
    print(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Extract latent representations (needed to determine scaler)
    print("Extracting latent representations...")
    with torch.no_grad():
        z_train = extract_latent_representations(model, X_train_scaled, device)
        z_test = extract_latent_representations(model, X_test_scaled, device)

    scaler_z = StandardScaler()
    z_train_scaled = scaler_z.fit_transform(z_train)
    z_test_scaled = scaler_z.transform(z_test)

    # Create degradation analyzer and compute progression data
    print("Computing progression data with real backbone features...")
    analyzer = DegradationAnalyzer(
        model, dataset_train, scaler, device,
        scaler_z=scaler_z, proto_size=args.latent_dim,
        window_size=args.window_size, scale=True
    )
    progression_scaler = analyzer.fit_progression_scaler(method='angular')

    all_data_real = analyzer.compute_all_progression_data(method='angular')

    analyzer_test = DegradationAnalyzer(
        model, dataset_test, scaler, device,
        scaler_z=scaler_z, proto_size=args.latent_dim,
        window_size=args.window_size,
        progression_scaler=progression_scaler, scale=True
    )
    all_data_test_real = analyzer_test.compute_all_progression_data(method='angular')

    print(f"Real data - Train units: {len(all_data_real)}, Test units: {len(all_data_test_real)}")

    # Test with real features (baseline)
    print("\n" + "="*80)
    print("BASELINE: Testing with REAL features")
    print("="*80)
    feature_group = ['z_latent', 'angles']
    print(f"Features: {feature_group}")

    try:
        baseline_results = benchmark_model(
            all_data_real, all_data_test_real, dataset_test_RUL,
            feature_names=feature_group,
            model_type='transformer',
            smoothing_window=1,
            transformer_epochs=20,
            transformer_lr=0.001,
            dataset_name=f"{args.dataset}_BASELINE",
            sheet_name='timestep_memorization_test',
            record_benchmark=False
        )
        print("✓ Baseline completed successfully")
    except Exception as e:
        print(f"✗ Baseline failed: {e}")
        baseline_results = None

    # Create timestep-only data
    print("\n" + "="*80)
    print("TEST: Replacing all features with TIMESTEP INDICES")
    print("="*80)

    all_data_timesteps = create_timestep_data(all_data_real)
    all_data_test_timesteps = create_timestep_data(all_data_test_real)

    # Test with timestep features
    print(f"Features: {feature_group} (but replaced with timesteps)")
    print(f"Timestep data - Train units: {len(all_data_timesteps)}, Test units: {len(all_data_test_timesteps)}")

    try:
        timestep_results = benchmark_model(
            all_data_timesteps, all_data_test_timesteps, dataset_test_RUL,
            feature_names=feature_group,
            model_type='transformer',
            smoothing_window=1,
            transformer_epochs=20,
            transformer_lr=0.001,
            dataset_name=f"{args.dataset}_TIMESTEPS",
            sheet_name='timestep_memorization_test',
            record_benchmark=False
        )
        print("✓ Timestep test completed successfully")
    except Exception as e:
        print(f"✗ Timestep test failed: {e}")
        timestep_results = None

    # Print results comparison
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    if baseline_results and timestep_results:
        _, y_baseline_pred, baseline_metrics = baseline_results
        _, y_timestep_pred, timestep_metrics = timestep_results

        baseline_rmse = baseline_metrics.get('rmse', float('inf'))
        timestep_rmse = timestep_metrics.get('rmse', float('inf'))
        baseline_r2 = baseline_metrics.get('r2', -float('inf'))
        timestep_r2 = timestep_metrics.get('r2', -float('inf'))

        print(f"\nBaseline (Real Features):")
        print(f"  RMSE: {baseline_rmse:.4f}")
        print(f"  R²:   {baseline_r2:.4f}")

        print(f"\nTimestep Features:")
        print(f"  RMSE: {timestep_rmse:.4f}")
        print(f"  R²:   {timestep_r2:.4f}")

        rmse_degradation = ((timestep_rmse - baseline_rmse) / baseline_rmse) * 100
        r2_degradation = ((baseline_r2 - timestep_r2) / abs(baseline_r2)) * 100 if baseline_r2 != 0 else 0

        print(f"\nDegradation:")
        print(f"  RMSE change: {rmse_degradation:+.2f}%")
        print(f"  R² change:   {r2_degradation:+.2f}%")

        if timestep_rmse < baseline_rmse * 1.5:
            print("\n⚠️  WARNING: Timestep features perform surprisingly well!")
            print("    This suggests the backbone may be memorizing timesteps.")
        else:
            print("\n✓ Timestep features perform much worse.")
            print("    This is expected - the backbone learned meaningful state representations.")
    else:
        print("Could not compare results - one or both tests failed")

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)


if __name__ == "__main__":
    main()
