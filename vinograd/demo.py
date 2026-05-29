"""
STEP (Spatial-Temporal Embedding Prototype) Demo
Official demo for ICML paper: "Prototypical Trajectories in Latent Space for Neural Time Series Analysis"

This demo demonstrates the STEP approach on neural time series data, showing:
1. Data loading and preprocessing
2. Model training with triplet loss and prototype learning
3. Latent space visualization
4. Reconstruction error analysis
"""

import os
import sys
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.auto_encoders import SimpleTransformerTriplet
from vinograd.dataset import TripletWindowDataset
from losses.losses import combined_loss


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    # Ensure seed is within valid range for numpy (0 to 2**32 - 1)
    numpy_seed = seed % (2**32)
    
    torch.manual_seed(seed)
    np.random.seed(numpy_seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_and_preprocess_data():
    """Load and preprocess neural data"""
    try:
        import lindi
        import jax.numpy as jnp
        
        print("Loading neural data from DANDI archive...")
        url = 'https://api.dandiarchive.org/api/assets/1e11c74e-6f25-4604-9216-5b861fec8f1c/download/'
        
        # Load the remote file
        f = lindi.LindiH5pyFile.from_hdf5_file(url)
        
        # Load the neurodata object (TimeSeries)
        NeuralTrace = f['/processing/ophys/NeuralTrace']
        neural = NeuralTrace['data'][:]
        rate = 10  # 10 Hz
        
        print(f"shape of neural activity: {neural.shape}")
        n_timesteps, n_neurons = neural.shape
        t_max = n_timesteps / rate
        t_grid = jnp.linspace(0, t_max, n_timesteps)
        dt = 1. / rate  # seconds per bin
        n_trials = 1
        
        print(f"trial duration: {t_max}")
        print(f"seconds per bin: {dt}")
        
        # Normalize data
        norm_neural = (neural - neural.mean(0)) / neural.std(0)
        
        # Set input times
        input1_times = jnp.array([74, 80])  # time interval (in seconds) of 1st intruder entrance
        input2_times = jnp.array([207, 212])  # time interval (in seconds) of 2nd intruder entrance
        
        # Convert to indices
        input1_inds = (input1_times * rate).astype(int)
        input2_inds = (input2_times * rate).astype(int)
        
        return norm_neural, input1_inds, input2_inds, rate
        
    except ImportError as e:
        print(f"Required packages not available: {e}")
        print("Please install lindi and jax: pip install lindi jax jaxlib")
        sys.exit(1)


def create_windows(data, window_size=30, step=1, threshold=-1, dataset_test_RUL=None, 
                   forecasting=False, last_only=False):
    """Create windows with unit IDs and RUL labels, and optionally subsequent windows for forecasting"""
    windows = []
    next_windows = [] if forecasting else None
    rul_labels = []
    unit_ids = []
    
    unique_units = data.keys()
    
    for unit_id in unique_units:
        unit_data = data[unit_id]
        n_samples = len(unit_data)
        
        # Adjust range based on whether we're doing forecasting
        if forecasting:
            max_i = n_samples - window_size - 1  # -1 to ensure we have a next window
        else:
            max_i = n_samples - window_size + 1
        
        for i in range(0, max_i, step):
            target_rul = n_samples - (i + window_size)
            
            # Adjust RUL if dataset_test_RUL is provided
            if dataset_test_RUL is not None:
                try:
                    if isinstance(unit_id, str) and unit_id.isdigit():
                        target_rul += dataset_test_RUL[int(unit_id)-1]
                    else:
                        target_rul += dataset_test_RUL[unit_id-1]
                except (IndexError, KeyError, TypeError):
                    pass
            
            # Apply threshold filter
            if threshold < 0 or target_rul > threshold:
                window = unit_data[i:i+window_size,].flatten()
                windows.append(window)
                rul_labels.append(target_rul)
                unit_ids.append(unit_id)
                
                if forecasting:
                    if last_only:
                        next_window = unit_data[i+1+window_size,].flatten()
                    else:
                        next_window = unit_data[i+1:i+1+window_size,].flatten()
                    next_windows.append(next_window)
    
    windows_array = np.array(windows)
    rul_labels_array = np.array(rul_labels)
    unit_ids_array = np.array(unit_ids)
    
    if forecasting:
        next_windows_array = np.array(next_windows)
        return windows_array, next_windows_array, rul_labels_array, unit_ids_array
    else:
        return windows_array, rul_labels_array, unit_ids_array


def train_epoch(model, dataloader, optimizer, device, alpha=0.5, beta=0.2, margin=1.0):
    """Optimized training epoch"""
    model.train()
    total_loss = total_recon_loss = total_trip_loss = total_proto_loss = 0
    
    for anchor, positive, negative, anchor_rul, is_initial, is_failure, next_state in dataloader:
        anchor = anchor.float().to(device)
        positive = positive.float().to(device)
        negative = negative.float().to(device)
        next_state = next_state.float().to(device)
        
        if not torch.is_tensor(is_initial):
            is_initial = torch.tensor(is_initial, device=device).float()
        else:
            is_initial = is_initial.to(device).float()
        
        if not torch.is_tensor(is_failure):
            is_failure = torch.tensor(is_failure, device=device).float()
        else:
            is_failure = is_failure.to(device).float()
        
        optimizer.zero_grad()
        
        # Single forward pass for anchor
        recon_anchor, z_anchor = model(anchor)
        
        # Compute embeddings for positive and negative in single batch
        positive_negative_batch = torch.cat([positive, negative], dim=0)
        with torch.no_grad():
            _, pn_embeddings = model(positive_negative_batch)
            z_positive, z_negative = torch.chunk(pn_embeddings, 2, dim=0)
        
        # Compute loss
        loss, recon_loss, trip_loss, proto_loss = combined_loss(
            z_anchor, z_positive, z_negative, recon_anchor, next_state,
            z_anchor, is_initial, is_failure, model, alpha, beta, margin
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_trip_loss += trip_loss.item()
        total_proto_loss += proto_loss.item()
    
    n_batches = len(dataloader)
    return (total_loss / n_batches, total_recon_loss / n_batches,
            total_trip_loss / n_batches, total_proto_loss / n_batches)


def evaluate_model(model, dataloader, device, alpha=0.5, beta=0.2, margin=1.0):
    """Optimized evaluation"""
    model.eval()
    total_loss = total_recon_loss = total_trip_loss = total_proto_loss = 0
    
    for anchor, positive, negative, anchor_rul, is_initial, is_failure in dataloader:
        anchor = anchor.float().to(device)
        positive = positive.float().to(device)
        negative = negative.float().to(device)
        
        if not torch.is_tensor(is_initial):
            is_initial = torch.tensor(is_initial, device=device).float()
        else:
            is_initial = is_initial.to(device).float()
        
        if not torch.is_tensor(is_failure):
            is_failure = torch.tensor(is_failure, device=device).float()
        else:
            is_failure = is_failure.to(device).float()
        
        # Forward passes
        recon_anchor, z_anchor = model(anchor)
        _, z_positive = model(positive)
        _, z_negative = model(negative)
        
        # Compute loss
        loss, recon_loss, trip_loss, proto_loss = combined_loss(
            z_anchor, z_positive, z_negative, recon_anchor, anchor,
            z_anchor, is_initial, is_failure, model, alpha, beta, margin
        )
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_trip_loss += trip_loss.item()
        total_proto_loss += proto_loss.item()
    
    n_batches = len(dataloader)
    return (total_loss / n_batches, total_recon_loss / n_batches,
            total_trip_loss / n_batches, total_proto_loss / n_batches)


def plot_training_losses(train_losses, train_recon_losses, train_trip_losses, train_proto_losses):
    """Plot training losses with improved visualization for ICML paper"""
    plt.figure(figsize=(16, 6))
    
    # Total Loss
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, 'b-', linewidth=2, label='Total Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Total Training Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Reconstruction Loss
    plt.subplot(1, 4, 2)
    plt.plot(train_recon_losses, 'g-', linewidth=2, label='Reconstruction Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Reconstruction Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Triplet Loss
    plt.subplot(1, 4, 3)
    plt.plot(train_trip_losses, 'r-', linewidth=2, label='Triplet Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Triplet Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Proto Loss
    plt.subplot(1, 4, 4)
    plt.plot(train_proto_losses, 'm-', linewidth=2, label='Prototype Loss')  # 'm' for magenta
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Prototype Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('STEP Training Loss Components', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_prototype_evolution(model, dataloader, device, num_samples=100):
    """Plot the evolution of embeddings toward prototypes during training"""
    model.eval()
    
    # Collect some embeddings
    embeddings = []
    is_initial_flags = []
    is_failure_flags = []
    
    with torch.no_grad():
        for i, (anchor, positive, negative, anchor_rul, is_initial, is_failure, next_state) in enumerate(dataloader):
            if i >= num_samples // dataloader.batch_size:
                break
            
            anchor = anchor.float().to(device)
            recon_anchor, z_anchor = model(anchor)
            
            embeddings.append(z_anchor.cpu().numpy())
            is_initial_flags.extend([1 if x else 0 for x in is_initial])
            is_failure_flags.extend([1 if x else 0 for x in is_failure])
    
    if embeddings:
        embeddings = np.concatenate(embeddings, axis=0)
        is_initial_flags = np.array(is_initial_flags)
        is_failure_flags = np.array(is_failure_flags)
        
        plt.figure(figsize=(12, 8))
        
        # Plot all embeddings
        plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                    c='lightgray', alpha=0.3, s=20, label='All embeddings')
        
        # Plot initial prototypes
        initial_embeddings = embeddings[is_initial_flags == 1]
        if len(initial_embeddings) > 0:
            plt.scatter(initial_embeddings[:, 0], initial_embeddings[:, 1],
                        c='green', s=100, label='Initial prototypes', alpha=0.7)
        
        # Plot failure prototypes
        failure_embeddings = embeddings[is_failure_flags == 1]
        if len(failure_embeddings) > 0:
            plt.scatter(failure_embeddings[:, 0], failure_embeddings[:, 1],
                        c='red', s=100, label='Failure prototypes', alpha=0.7)
        
        # Plot prototype vectors
        initial_proto = model.initial_prototype.cpu().numpy()
        failure_proto = model.failure_prototype.cpu().numpy()
        
        plt.scatter([initial_proto[0]], [initial_proto[1]], 
                    c='green', s=200, marker='*', edgecolor='black', linewidth=2, label='Initial prototype target')
        plt.scatter([failure_proto[0]], [failure_proto[1]],
                    c='red', s=200, marker='*', edgecolor='black', linewidth=2, label='Failure prototype target')
        
        plt.title('Embedding Space with Prototype Learning', fontsize=16)
        plt.xlabel(f'Latent Dimension 1', fontsize=12)
        plt.ylabel(f'Latent Dimension 2', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main(num_epochs=10, test_run=False):
    """Main demo function"""
    print("="*60)
    print("STEP Demo - ICML Paper Implementation")
    print("="*60)
    
    # Set random seeds for reproducibility
    set_random_seeds(16976296098443334824)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load and preprocess data
    norm_neural, input1_inds, input2_inds, rate = load_and_preprocess_data()
    
    # Training parameters
    window_size = 10
    step_size = 1
    forecasting = True
    last_only = False
    
    # Create windows
    neural_dict = {0: norm_neural}
    X_train, X_train_next, train_rul_labels, train_unit_ids = create_windows(
        neural_dict, window_size=window_size, step=step_size, 
        forecasting=forecasting, last_only=last_only
    )
    print(f'X_train: {X_train.shape}, X_train_next: {X_train_next.shape}, '
          f'train_rul_labels: {train_rul_labels.shape}, train_unit_ids: {train_unit_ids.shape}')
    
    # Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if last_only:
        scaler_next = MinMaxScaler()
        X_train_next_scaled = scaler_next.fit_transform(X_train_next)
    else:
        scaler_next = scaler
        X_train_next_scaled = scaler_next.transform(X_train_next)
    
    # Create dataset
    triplet_dataset = TripletWindowDataset(
        X_train_scaled, train_rul_labels, train_unit_ids, step_size=step_size,
        next_windows=X_train_next_scaled,
        margin=10, health_margin=30, n_initial_prototypes=5, n_failure_prototypes=5
    )
    
    # Create dataloader
    batch_size = 256 if not test_run else 32  # Smaller batch for test runs
    num_workers = 64 if device == 'cuda' else 0
    
    triplet_loader = DataLoader(
        triplet_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    # Initialize model with proto_size parameter
    input_dim = X_train_scaled.shape[1]
    output_dim = X_train_next.shape[1]
    latent_dim = 4
    proto_size = 4  # This will match the latent_dim
    
    print(f"Input dim: {input_dim}, Output dim: {output_dim}, Latent dim: {latent_dim}, Proto_size: {proto_size}")
    
    model = SimpleTransformerTriplet(
        input_dim, output_dim=output_dim, latent_dim=latent_dim, 
        d_model=32, nhead=8, num_layers=4, dropout=0.1, proto_size=proto_size
    ).to(device)
    
    print(f"Prototypes - Initial: {model.initial_prototype}, Failure: {model.failure_prototype}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    # Training configuration
    alpha = 1.0  # Triplet loss weight
    beta = 1.0   # Prototype loss weight
    margin = 10.0
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # Track losses
    train_losses, train_recon_losses, train_trip_losses, train_proto_losses = [], [], [], []
    best_test_loss = float('inf')
    best_model_wts = None
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        current_beta = max(0.3 * (0.95 ** epoch), 0.05)
        
        # Training
        train_metrics = train_epoch(model, triplet_loader, optimizer, device, alpha, current_beta, margin)
        avg_train_loss, avg_train_recon, avg_train_trip, avg_train_proto = train_metrics
        
        # Store metrics
        train_losses.append(avg_train_loss)
        train_recon_losses.append(avg_train_recon)
        train_trip_losses.append(avg_train_trip)
        train_proto_losses.append(avg_train_proto)
        
        scheduler.step()
        
        # Save best model
        if avg_train_loss < best_test_loss:
            best_test_loss = avg_train_loss
            best_model_wts = model.state_dict().copy()
            print(f'>>> Best model at epoch {epoch+1} (Test Loss: {best_test_loss:.4f})')
        
        # Print progress
        if (epoch + 1) % max(1, num_epochs // 5) == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Recon: {avg_train_recon:.4f} | '
                  f'Trip: {avg_train_trip:.4f} | '
                  f'Proto: {avg_train_proto:.4f}')
    
    # Training complete
    elapsed_time = time.time() - start_time
    print(f'\nTraining complete. Best Test Loss: {best_test_loss:.4f}')
    print(f'Training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')
    
    # Plot losses (always show for demo purposes, but skip for automated tests)
    if not test_run or num_epochs <= 10:  # Show plots for short runs
        try:
            plot_training_losses(train_losses, train_recon_losses, train_trip_losses, train_proto_losses)
            print("✓ Training loss plots displayed successfully")
            
            # Also show prototype evolution plot
            if len(triplet_loader) > 0:
                plot_prototype_evolution(model, triplet_loader, device)
                print("✓ Prototype evolution plot displayed successfully")
                
        except Exception as e:
            print(f"⚠ Could not display plots: {e}")
            print("  (This is normal in non-interactive environments)")
    
    # Return trained model and data for further analysis
    return model, X_train_scaled, X_train_next_scaled, scaler, input1_inds, input2_inds, step_size


def test_plotting():
    """Test that plotting functionality works"""
    print("Testing plotting functionality...")
    
    try:
        # Create some dummy data for testing plots
        dummy_losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        dummy_recon = [0.5, 0.4, 0.3, 0.2, 0.1]
        dummy_trip = [0.3, 0.25, 0.2, 0.15, 0.1]
        dummy_proto = [0.2, 0.15, 0.1, 0.05, 0.0]
        
        plot_training_losses(dummy_losses, dummy_recon, dummy_trip, dummy_proto)
        print("✓ Plotting functionality test successful!")
        return True
        
    except Exception as e:
        print(f"⚠ Plotting test failed: {e}")
        print("  (This might be expected in headless environments)")
        return False


if __name__ == "__main__":
    # Run demo with 10 epochs for testing
    print("Running STEP demo with 10 epochs...")
    model, X_train_scaled, X_train_next_scaled, scaler, input1_inds, input2_inds, step_size = main(num_epochs=100, test_run=True)
    print("Demo completed successfully!")
    
    # Test plotting separately
    test_plotting()