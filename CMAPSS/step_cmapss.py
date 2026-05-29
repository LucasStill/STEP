#!/usr/bin/env python3

import sys
import os

# Get the absolute path to the project root
# Handle both direct execution and execution from parent directory
if os.path.basename(os.getcwd()) == 'CMAPSS':
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
else:
    project_root = os.path.abspath(os.getcwd())

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add the CMAPSS directory to path for imports
cmapss_dir = os.path.join(project_root, 'CMAPSS')
if cmapss_dir not in sys.path:
    sys.path.insert(0, cmapss_dir)

import torch
from CMAPSS.dataset import create_windows
import numpy as np
from models.auto_encoders import SimpleTransformerTriplet
from torch.utils.data import DataLoader
from CMAPSS.dataset import TripletWindowDataset
from sklearn.preprocessing import MinMaxScaler
import argparse

from datetime import datetime



def parse_arguments(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='STEP CMAPSS Training Script')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='FD001', 
                       help='Dataset name (e.g., FD001, FD002, etc.)')
    parser.add_argument('--seed', type=int, default=16976296098443334824,
                       help='Random seed for reproducibility')
    
    # Model architecture parameters
    parser.add_argument('--latent_dim', type=int, default=16,
                       help='Latent dimension size')
    parser.add_argument('--d_model', type=int, default=32,
                       help='Transformer d_model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for optimizer')
    
    # Loss function parameters
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Triplet loss weight')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='Prototype loss weight')
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Margin for triplet loss')
    
    # Window and data parameters
    parser.add_argument('--window_size', type=int, default=10,
                       help='Window size for time series')
    parser.add_argument('--R_early_train', type=int, default=0,
                       help='Early RUL threshold for training')
    
    # Dataset parameters for TripletWindowDataset
    parser.add_argument('--triplet_margin', type=int, default=10,
                       help='Margin for triplet dataset')
    parser.add_argument('--health_margin', type=int, default=20,
                       help='Health margin for triplet dataset')
    parser.add_argument('--n_initial_prototypes', type=int, default=8,
                       help='Number of initial prototypes')
    parser.add_argument('--n_failure_prototypes', type=int, default=8,
                       help='Number of failure prototypes')
    
    # Output and plotting options
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save model and results')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Custom model name (without extension)')
    parser.add_argument('--no_plotting', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--no_benchmark', action='store_true',
                       help='Disable downstream benchmarking')
    
    return parser.parse_args(args)

def main():
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)  # numpy uses 32-bit seeds
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data - try multiple possible paths
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
        raise FileNotFoundError(f"Could not find dataset files for {args.dataset} in any of: {data_paths}")
    
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
        train_data, window_size=args.window_size, threshold=args.R_early_train
    )

    X_test, test_rul_labels, test_unit_ids = create_windows(
        test_data, window_size=args.window_size, threshold=args.R_early_train,
        dataset_test_RUL=dataset_test_RUL
    )

    # Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create datasets with prototype support
    triplet_dataset = TripletWindowDataset(
        X_train_scaled, train_rul_labels, train_unit_ids,
        margin=args.triplet_margin, health_margin=args.health_margin, 
        n_initial_prototypes=args.n_initial_prototypes, 
        n_failure_prototypes=args.n_failure_prototypes
    )

    triplet_dataset_test = TripletWindowDataset(
        X_test_scaled, test_rul_labels, test_unit_ids,
        margin=args.triplet_margin, health_margin=args.health_margin,
        n_initial_prototypes=args.n_initial_prototypes, n_failure_prototypes=0
    )

    # Create dataloaders with multiple workers
    num_workers = 64 if device == 'cuda' else 0

    triplet_loader = DataLoader(
        triplet_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_triplet_loader = DataLoader(
        triplet_dataset_test, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Initialize model
    input_dim = X_train_scaled.shape[1]
    model = SimpleTransformerTriplet(
        input_dim, 
        latent_dim=args.latent_dim, 
        d_model=args.d_model, 
        nhead=args.nhead, 
        num_layers=args.num_layers, 
        dropout=args.dropout
    ).to(device)

    # Use more efficient optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    # Training configuration
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Track losses
    train_losses, train_recon_losses, train_trip_losses, train_proto_losses = [], [], [], []
    test_losses, test_recon_losses, test_trip_losses, test_proto_losses = [], [], [], []

    best_test_loss = float('inf')
    best_model_wts = None

    from models.auto_encoders import train_epoch, evaluate_model

    print("Starting training...")
    for epoch in range(args.epochs):

        current_beta = max(args.beta * (0.95 ** epoch), 0.05)

        # Training
        train_metrics = train_epoch(model, triplet_loader, optimizer, device, args.alpha, current_beta, args.margin)
        avg_train_loss, avg_train_recon, avg_train_trip, avg_train_proto = train_metrics

        # Evaluation
        test_metrics = evaluate_model(model, test_triplet_loader, device, args.alpha, current_beta, args.margin)
        avg_test_loss, avg_test_recon, avg_test_trip, avg_test_proto = test_metrics

        # Store metrics
        train_losses.append(avg_train_loss)
        train_recon_losses.append(avg_train_recon)
        train_trip_losses.append(avg_train_trip)
        train_proto_losses.append(avg_train_proto)

        test_losses.append(avg_test_loss)
        test_recon_losses.append(avg_test_recon)
        test_trip_losses.append(avg_test_trip)
        test_proto_losses.append(avg_test_proto)

        scheduler.step()

        # Save best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_wts = model.state_dict().copy()
            print(f'>>> Best model at epoch {epoch+1} (Test Loss: {best_test_loss:.4f})')

        # Print progress
        if (epoch + 1) % 20 == 0 or True:
            print(f'Epoch {epoch+1}/{args.epochs}: '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Recon: {avg_train_recon:.4f} | '
                  f'Trip: {avg_train_trip:.4f} | '
                  f'Proto: {avg_train_proto:.4f} || '
                  f'Test Loss: {avg_test_loss:.4f} | '
                  f'Recon: {avg_test_recon:.4f} | '
                  f'Trip: {avg_test_trip:.4f} | '
                  f'Proto: {avg_test_proto:.4f}')

    # Restore best model
    #model.load_state_dict(best_model_wts)
    print(f'\nTraining complete. Best Test Loss: {best_test_loss:.4f}')

    # Generate model filename with hyperparameters
    if args.model_name:
        model_filename = f"{args.model_name}.pth"
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{args.dataset}_z{args.latent_dim}_w{args.window_size}_a{args.alpha}_b{args.beta}_e{args.epochs}_{timestamp}.pth"
    
    model_path = os.path.join(args.output_dir, model_filename)
    
    # Save model state dict
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Save additional training metadata
    metadata = {
        'dataset': args.dataset,
        'latent_dim': args.latent_dim,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'alpha': args.alpha,
        'beta': args.beta,
        'margin': args.margin,
        'window_size': args.window_size,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_recon_losses': train_recon_losses,
        'test_recon_losses': test_recon_losses,
        'train_trip_losses': train_trip_losses,
        'test_trip_losses': test_trip_losses,
        'train_proto_losses': train_proto_losses,
        'test_proto_losses': test_proto_losses,
        'best_test_loss': best_test_loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save metadata as JSON
    metadata_filename = model_filename.replace('.pth', '_metadata.json')
    metadata_path = os.path.join(args.output_dir, metadata_filename)
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training metadata saved to: {metadata_path}")
    
    model.eval()
    
    # Store backbone model name for downstream benchmarking
    backbone_model_name = model_filename.replace('.pth', '')

    plotting = not args.no_plotting

    if plotting:
        import matplotlib.pyplot as plt

        # Plotting the losses
        plt.figure(figsize=(15, 5))

        # Total Loss
        plt.subplot(1, 4, 1)
        plt.plot(train_losses, label='Train Total Loss')
        plt.plot(test_losses, label='Test Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Total Loss')
        plt.legend()
        plt.grid(True)

        # Reconstruction Loss
        plt.subplot(1, 4, 2)
        plt.plot(train_recon_losses, label='Train Recon Loss')
        plt.plot(test_recon_losses, label='Test Recon Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Reconstruction Loss')
        plt.legend()
        plt.grid(True)

        # Triplet Loss
        plt.subplot(1, 4, 3)
        plt.plot(train_trip_losses, label='Train Triplet Loss')
        plt.plot(test_trip_losses, label='Test Triplet Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Triplet Loss')
        plt.title('Triplet Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 4, 4)
        plt.plot(train_proto_losses, label='Train Proto Loss')
        plt.plot(test_proto_losses, label='Test Proto Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Proto Loss')
        plt.title('Proto Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        figures_dir = os.path.join(project_root, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, f'{args.dataset}_z{args.latent_dim}_losses.pdf'), 
                   facecolor='white', bbox_inches='tight', dpi=200)
        plt.close()

    from sklearn.preprocessing import StandardScaler
    from latent_his.extract_latent import extract_latent_representations

    model.to(device)

    # Usage with your transformer model:
    model.eval()
    with torch.no_grad():
        # Get latent vectors for training data
        z_train = extract_latent_representations(model, X_train_scaled, device)

        # Get latent vectors for test data
        z_test = extract_latent_representations(model, X_test_scaled, device)

    # Scale latent vectors
    scaler_z = StandardScaler()
    z_train_scaled = scaler_z.fit_transform(z_train)
    z_test_scaled = scaler_z.transform(z_test)

    print(f"Train embeddings shape: {z_train.shape}")
    print(f"Test embeddings shape: {z_test.shape}")

    from CMAPSS.dataset import process_dataset

    # Create windows WITHOUT RUL as feature
    # pick a unit_id to visualize and sample it
    unit_id = 2
    sample = process_dataset(dataset_train, unit_id=unit_id)

    # Create windows for trajectory (same as training)
    traj_windows, _, _ = create_windows(
        sample,
        window_size=args.window_size,
        threshold=-1  # Include all points
    )

    # Scale using SAME scaler
    traj_windows_scaled = scaler.transform(traj_windows)

    # Get latent vectors for trajectory
    with torch.no_grad():
        z_traj = extract_latent_representations(model, traj_windows_scaled, device)
    z_traj_scaled = scaler_z.transform(z_traj)

    # Plot trajectory in latent space
    if plotting:
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        ax.set_facecolor('white')
        ax.grid(True, color='lightgray')
        ax.tick_params(axis='both', colors='black')

        # Plot training data as background
        plt.scatter(z_train_scaled[:, 0], z_train_scaled[:, 1],
                    c='lightgray', alpha=0.1, s=5, label='Training Data')

        # Plot trajectory with color gradient
        scatter = plt.scatter(z_traj_scaled[:, 0], z_traj_scaled[:, 1],
                    c=np.arange(len(z_traj_scaled)),
                    cmap='viridis', s=50, edgecolor='k',
                    label=f'Unit {unit_id} Trajectory')

        # Connect points with lines
        plt.plot(z_traj_scaled[:, 0], z_traj_scaled[:, 1],
                 'b-', alpha=0.3, linewidth=1)

        # Add markers
        plt.scatter(z_traj_scaled[0, 0], z_traj_scaled[0, 1],
                    c='green', s=200, marker='o', edgecolor='k',
                    label='Start (Healthy)')
        plt.scatter(z_traj_scaled[-1, 0], z_traj_scaled[-1, 1],
                    c='red', s=200, marker='X', edgecolor='k',
                    label='End (Failure)')

        # Add direction arrow
        if len(z_traj_scaled) > 1:
            mid_point = len(z_traj_scaled) // 2
            plt.annotate('',
                        xytext=(z_traj_scaled[mid_point-1, 0], z_traj_scaled[mid_point-1, 1]),
                        xy=(z_traj_scaled[mid_point, 0], z_traj_scaled[mid_point, 1]),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5, alpha=0.8))

        plt.colorbar(scatter, label='Time Step in Trajectory')
        plt.title(f'Degradation Path of Unit {unit_id} in Latent Space')
        plt.xlabel('Latent Dimension 1 (scaled)')
        plt.ylabel('Latent Dimension 2 (scaled)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'{args.dataset}_z{args.latent_dim}_STEP.pdf'), 
                   facecolor='white', bbox_inches='tight', dpi=200)
        plt.close()

        # Reconstruction error plot
        model.eval()
        with torch.no_grad():
            recon = model(torch.FloatTensor(traj_windows_scaled).to(device))[0].cpu().numpy()
        errors = np.linalg.norm(abs(traj_windows_scaled - recon), axis=1)

        plt.figure(figsize=(12, 6))
        plt.plot(errors, 'r-')
        plt.title(f'Reconstruction Error for Unit {unit_id}')
        plt.xlabel('Time Step')
        plt.ylabel('Error')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'{args.dataset}_z{args.latent_dim}_recon_error.pdf'), 
                   facecolor='white', bbox_inches='tight', dpi=200)
        plt.close()

    # Downstream task evaluation
    if not args.no_benchmark:
        from CMAPSS.degrad_analyzer import DegradationAnalyzer

        # Initialize once
        analyzer = DegradationAnalyzer(model, dataset_train, scaler, device, scaler_z=scaler_z, 
                                       proto_size=args.latent_dim, window_size=args.window_size, scale=True)
        progression_scaler = analyzer.fit_progression_scaler(method='angular')

        # Compute for all units (no plots)
        all_data = analyzer.compute_all_progression_data(method='angular')

        analyzer_test = DegradationAnalyzer(model, dataset_test, scaler, device, scaler_z=scaler_z, 
                                            proto_size=args.latent_dim, window_size=args.window_size, 
                                            progression_scaler=progression_scaler, scale=True)
        all_data_test = analyzer_test.compute_all_progression_data(method='angular')


        # Automate by choosing different indicators
        from models.downstream_head import benchmark_model

        feature_groups = [
            ['z_latent', 'angles'], 
            ['z_latent'], 
            ['progression_scores', 'angles_deg'], 
            ['progression_scores', 'angles_deg', 'radii'], 
            ['progression_scores', 'angles_deg', 'radii', 'z_latent'], 
            ['angles'], 
            ['angles', 'radii']
        ]
        
        for i, feature_group in enumerate(feature_groups):
            print(f'Doing run {i+1}/{len(feature_groups)} for {feature_group}')
            transformer_results = benchmark_model(
                all_data, all_data_test, dataset_test_RUL,
                feature_names=feature_group,
                model_type='transformer',
                smoothing_window=1,
                transformer_epochs=20,
                transformer_lr=0.001,
                dataset_name=args.dataset,
                sheet_name='benchmarks',
                record_benchmark=True,
                backbone_model_name=backbone_model_name,
                backbone_latent_dim=args.latent_dim,
                backbone_alpha=args.alpha,
                backbone_beta=args.beta,
                backbone_epochs=args.epochs
            )

    print("Training and evaluation completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Plots saved to: {os.path.join(project_root, 'figures')}")
    print(f"Benchmark results recorded to benchmarks/benchmarks.csv")

if __name__ == "__main__":
    main()