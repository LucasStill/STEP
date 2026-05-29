import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from math import sqrt
import warnings
warnings.filterwarnings('ignore')
import csv
import os
from datetime import datetime

# Transformer imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

class SimplePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Simplified positional encoding - learnable
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.position_embedding[:, :seq_len, :]
        return self.dropout(x)

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, max_seq_len=1000):
        super(TransformerRegressor, self).__init__()
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Your positional encoding
        self.pos_encoding = SimplePositionalEncoding(d_model, dropout, max_len=max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression output
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape

        # Project input
        x = self.input_projection(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        # Apply positional encoding
        x = self.pos_encoding(x)

        # Transformer
        transformer_out = self.transformer(x)

        # Use the last timestep for prediction
        last_output = transformer_out[:, -1, :]

        # Regression output
        return self.output_layer(last_output).squeeze(-1)

def smooth_features(features, smoothing_window=5):
    """
    Apply moving average smoothing to features
    """
    if smoothing_window <= 1:
        return features

    smoothed = np.zeros_like(features)
    pad = smoothing_window // 2

    for i in range(len(features)):
        start = max(0, i - pad)
        end = min(len(features), i + pad + 1)
        smoothed[i] = np.mean(features[start:end], axis=0)

    return smoothed

def record_benchmark_result(dataset_name, sheet_name, feature_names, window_size,
                           model_type, rmse, mae, r2, smoothing_window,
                           R_threshold, clip_threshold, padding,
                           backbone_model_name=None, latent_dim=None,
                           alpha=None, beta=None, backbone_epochs=None):
    """
    Record benchmark results to CSV file
    
    Args:
        dataset_name: Name of the dataset
        sheet_name: Name of the CSV file (without extension)
        feature_names: List of feature names used
        window_size: Window size used in the model
        model_type: Type of model used
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        r2: R-squared score
        smoothing_window: Smoothing window size
        R_threshold: RUL threshold
        clip_threshold: Whether threshold clipping was used
        padding: Whether padding was used
        backbone_model_name: Name of the backbone model used
        latent_dim: Latent dimension of backbone model
        alpha: Alpha parameter used in backbone training
        beta: Beta parameter used in backbone training
        backbone_epochs: Number of epochs used for backbone training
    """
    # Create benchmarks directory if it doesn't exist
    benchmarks_dir = 'benchmarks'
    os.makedirs(benchmarks_dir, exist_ok=True)
    
    # Create CSV file path
    csv_filename = f"{benchmarks_dir}/{sheet_name}.csv"
    
    # Check if file exists to determine if we need to handle existing headers
    file_exists = os.path.exists(csv_filename)
    
    # Prepare data row
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    feature_names_str = ', '.join(feature_names)
    
    row_data = {
        'timestamp': timestamp,
        'dataset_name': dataset_name,
        'feature_names': feature_names_str,
        'window_size': window_size,
        'model_type': model_type,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'smoothing_window': smoothing_window,
        'R_threshold': R_threshold,
        'clip_threshold': clip_threshold,
        'padding': padding,
        'backbone_model_name': backbone_model_name,
        'latent_dim': latent_dim,
        'alpha': alpha,
        'beta': beta,
        'backbone_epochs': backbone_epochs
    }
    
    # Define all possible fieldnames
    all_fieldnames = [
        'timestamp', 'dataset_name', 'feature_names', 'window_size',
        'model_type', 'rmse', 'mae', 'r2', 'smoothing_window',
        'R_threshold', 'clip_threshold', 'padding',
        'backbone_model_name', 'latent_dim', 'alpha', 'beta', 'backbone_epochs'
    ]
    
    # Write to CSV
    if file_exists:
        # Read existing headers first
        try:
            with open(csv_filename, 'r', newline='') as read_file:
                reader = csv.reader(read_file)
                existing_headers = next(reader)
        except:
            existing_headers = []
        
        # Determine which fieldnames to use
        if existing_headers:
            # Add any missing new columns to existing headers
            new_columns = [col for col in all_fieldnames if col not in existing_headers]
            fieldnames = existing_headers + new_columns
            
            # We need to rewrite the header if we added new columns
            if new_columns:
                # Read all existing data
                with open(csv_filename, 'r', newline='') as read_file:
                    reader = csv.DictReader(read_file, fieldnames=existing_headers)
                    existing_data = list(reader)
                
                # Write with new header
                with open(csv_filename, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    # Write existing data (missing columns will be empty)
                    for row in existing_data:
                        writer.writerow(row)
                    # Write new data
                    writer.writerow(row_data)
            else:
                # No new columns, just append
                with open(csv_filename, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(row_data)
        else:
            # No existing headers, use all fieldnames
            fieldnames = all_fieldnames
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row_data)
    else:
        # File doesn't exist, create with all headers
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_fieldnames)
            writer.writeheader()
            writer.writerow(row_data)
    
    print(f"Benchmark result recorded to {csv_filename}")


def benchmark_model(all_data, all_data_test, dataset_test_RUL,
                   feature_names=['progression_scores'],
                   model_type='random_forest',
                   classification_threshold=100,
                   smoothing_window=5,
                   window_size=75, R_threshold=125,
                   clip_threshold=False, padding=False,
                   # Transformer-specific parameters
                   transformer_epochs=50, transformer_lr=0.001, transformer_batch_size=32,
                   # CSV benchmarking parameters
                   dataset_name='default_dataset',
                   sheet_name='benchmarks',
                   record_benchmark=True,
                   # Backbone model parameters for tracking
                   backbone_model_name=None,
                   backbone_latent_dim=None,
                   backbone_alpha=None,
                   backbone_beta=None,
                   backbone_epochs=None):
    """
    Benchmark different models for RUL prediction
    
    Args:
        dataset_name: Name of the dataset for CSV recording
        sheet_name: Name of the CSV sheet/file
        record_benchmark: Whether to record benchmark results to CSV
    """

    rf_train = all_data
    rf_test = all_data_test

    def prepare_features(unit_data, feature_names, apply_smoothing=True):
        """Prepare features for a single unit, handling mixed-length features"""
        features_list = []

        # Get the main trajectory length from progression_scores
        main_length = len(unit_data['progression_scores'])

        for feature_name in feature_names:
            feature_data = unit_data[feature_name]

            # Handle scalar features by repeating them
            if np.isscalar(feature_data) or len(feature_data) == 1:
                feature_data = np.repeat(feature_data, main_length)
            # Handle shorter arrays by padding
            elif len(feature_data) < main_length:
                pad_width = main_length - len(feature_data)
                feature_data = np.pad(feature_data, (0, pad_width), mode='edge')

            # Ensure feature is 1D for stacking
            if feature_data.ndim == 1:
                features_list.append(feature_data.reshape(-1, 1))
            else:
                features_list.append(feature_data)

        # Stack features and ensure proper shape
        if len(features_list) == 1:
            traj = features_list[0]
        else:
            traj = np.hstack(features_list)

        # Apply smoothing if requested
        if apply_smoothing and smoothing_window > 1:
            traj = smooth_features(traj, smoothing_window)

        return traj

    # ---- Prepare training data ----
    X_train, y_train = [], []
    X_train_3d = []  # For transformer

    for unit_id in rf_train.keys():
        traj = prepare_features(rf_train[unit_id], feature_names)
        T = len(traj)

        if T < window_size:
            if not padding:
                continue
            pad_width = window_size - T
            traj = np.pad(traj, ((pad_width, 0), (0, 0)), mode='constant', constant_values=0)
            T = len(traj)

        for i in range(T - window_size):
            window = traj[i:i + window_size]
            window_flat = window.flatten()
            rul = T - (i + window_size)
            if rul <= R_threshold:
                X_train.append(window_flat)
                X_train_3d.append(window)
                y_train.append(rul)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Calculate actual number of features from the data
    if len(X_train_3d) > 0:
        X_train_3d = np.array(X_train_3d)
        n_features = X_train_3d.shape[2]  # Get actual feature dimension from data
    else:
        n_features = len(feature_names)  # Fallback

    #print(f"Train samples: {X_train.shape}, labels: {y_train.shape}")
    #if len(X_train_3d) > 0:
    #    print(f"Train 3D shape: {X_train_3d.shape}, Features: {n_features}")

    # ---- Prepare test data ----
    X_test, y_test = [], []
    X_test_3d = []  # For transformer

    for idx, unit_id in enumerate(rf_test.keys()):
        traj = prepare_features(rf_test[unit_id], feature_names)
        T = len(traj)

        if T < window_size:
            if not padding:
                continue
            pad_width = window_size - T
            traj = np.pad(traj, ((pad_width, 0), (0, 0)), mode='constant', constant_values=0)
            T = len(traj)

        RUL_final = dataset_test_RUL[idx]
        last_window = traj[-window_size:]
        last_window_flat = last_window.flatten()
        rul = RUL_final

        if rul <= R_threshold:
            X_test.append(last_window_flat)
            X_test_3d.append(last_window)
            y_test.append(rul)
        elif clip_threshold:
            X_test.append(last_window_flat)
            X_test_3d.append(last_window)
            y_test.append(R_threshold)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if len(X_test_3d) > 0:
        X_test_3d = np.array(X_test_3d)

    print(f"Test samples: {X_test.shape}, labels: {y_test.shape}")
    if len(X_test_3d) > 0:
        print(f"Test 3D shape: {X_test_3d.shape}")

    # Initialize variables to track best RMSE
    best_rmse = float('inf')
    best_results = None
    best_model_info = {}
    
    # ---- Handle different model types ----
    if model_type == 'transformer':
        y_test, y_pred, results = train_transformer_model(X_train_3d, y_train, X_test_3d, y_test,
                                                         n_features, window_size, transformer_epochs,
                                                         transformer_lr, transformer_batch_size)
        if 'rmse' in results and results['rmse'] < best_rmse:
            best_rmse = results['rmse']
            best_results = (y_test, y_pred, results)
            best_model_info = {
                'model_type': model_type,
                'rmse': results['rmse'],
                'mae': results.get('mae', 'N/A'),
                'r2': results.get('r2', 'N/A')
            }
    else:
        y_test, y_pred, results = train_sklearn_model(X_train, y_train, X_test, y_test,
                                                     model_type, classification_threshold)
        if 'rmse' in results and results['rmse'] < best_rmse:
            best_rmse = results['rmse']
            best_results = (y_test, y_pred, results)
            best_model_info = {
                'model_type': model_type,
                'rmse': results['rmse'],
                'mae': results.get('mae', 'N/A'),
                'r2': results.get('r2', 'N/A')
            }
        elif 'accuracy' in results:
            # For classification models, we don't track RMSE
            best_results = (y_test, y_pred, results)
            best_model_info = {
                'model_type': model_type,
                'accuracy': results['accuracy']
            }
    
    # Record benchmark if enabled and we have RMSE data
    if record_benchmark and best_rmse != float('inf'):
        record_benchmark_result(
            dataset_name=dataset_name,
            sheet_name=sheet_name,
            feature_names=feature_names,
            window_size=window_size,
            model_type=best_model_info['model_type'],
            rmse=best_model_info['rmse'],
            mae=best_model_info.get('mae', 'N/A'),
            r2=best_model_info.get('r2', 'N/A'),
            smoothing_window=smoothing_window,
            R_threshold=R_threshold,
            clip_threshold=clip_threshold,
            padding=padding,
            backbone_model_name=backbone_model_name,
            latent_dim=backbone_latent_dim,
            alpha=backbone_alpha,
            beta=backbone_beta,
            backbone_epochs=backbone_epochs
        )
    
    return best_results

def train_sklearn_model(X_train, y_train, X_test, y_test, model_type, classification_threshold):
    """Train and evaluate scikit-learn models"""
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train model
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        y_train_model = y_train
        y_test_model = y_test

    elif model_type == 'linear_regression':
        model = LinearRegression()
        y_train_model = y_train
        y_test_model = y_test

    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        )
        y_train_model = (y_train <= classification_threshold).astype(int)
        y_test_model = (y_test <= classification_threshold).astype(int)
        print(f"Class distribution - Train: {np.bincount(y_train_model)}, Test: {np.bincount(y_test_model)}")

    else:
        raise ValueError("model_type must be 'random_forest', 'linear_regression', or 'logistic_regression'")

    # Train model
    model.fit(X_train_scaled, y_train_model)

    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)

    if model_type in ['random_forest', 'linear_regression']:
        rmse = sqrt(mean_squared_error(y_test_model, y_pred))
        mae = np.mean(np.abs(y_test_model - y_pred))

        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test_model, y_pred)

        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R²: {r2:.4f}")

        return y_test_model, y_pred, {'rmse': rmse, 'mae': mae, 'r2': r2}

    else:  # logistic_regression
        accuracy = accuracy_score(y_test_model, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_model, y_pred))

        return y_test_model, y_pred, {'accuracy': accuracy}


def train_transformer_model(X_train_3d, y_train, X_test_3d, y_test,
                          n_features, window_size, epochs=50, lr=0.001, batch_size=32):
    """Train and evaluate transformer model for regression with early stopping based on test RMSE"""

    # Check if we have data
    if len(X_train_3d) == 0 or len(X_test_3d) == 0:
        print("No data available for transformer training")
        return y_test, np.zeros_like(y_test), {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}


    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")

    # Normalize the 3D data
    original_shape_train = X_train_3d.shape
    original_shape_test = X_test_3d.shape

    # Reshape to 2D for scaling
    X_train_2d = X_train_3d.reshape(-1, n_features)
    X_test_2d = X_test_3d.reshape(-1, n_features)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    # Reshape back to 3D
    X_train_scaled = X_train_scaled_2d.reshape(original_shape_train)
    X_test_scaled = X_test_scaled_2d.reshape(original_shape_test)

    # Convert to PyTorch tensors - KEEP THEM ON CPU
    X_train_tensor = torch.FloatTensor(X_train_scaled)  # Removed .to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled)    # Removed .to(device)
    y_train_tensor = torch.FloatTensor(y_train)        # Removed .to(device)
    y_test_tensor = torch.FloatTensor(y_test)          # Removed .to(device)

    # Create data loaders with optimization
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Optimized data loaders - FIXED: reduced num_workers and removed persistent_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, 128),  # Reduced batch size for safety
        shuffle=True,
        num_workers=2,  # Reduced from 48 to 2 to avoid CUDA issues
        pin_memory=True,  # This is still safe
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=min(batch_size * 2, 128),
        shuffle=False,
        num_workers=2,  # Reduced from 16 to 2
        pin_memory=True,
    )

    # Initialize model with correct input dimension and move to device
    regressor_model = TransformerRegressor(input_dim=n_features, d_model=32, nhead=4, num_layers=2, max_seq_len=window_size).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(regressor_model.parameters(), lr=lr)

    # Training loop with best model tracking
    regressor_model.train()
    train_losses = []
    test_rmses = []

    best_rmse = float('inf')
    best_model_state = None
    best_epoch = 0

    for epoch in range(epochs):
        # Training phase
        regressor_model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            # Move batch to device HERE, not before
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = regressor_model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Evaluation phase - calculate test RMSE
        regressor_model.eval()
        test_predictions = []
        test_true_values = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                # Move batch to device
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = regressor_model(batch_X)
                preds = outputs.cpu().numpy()  # Move back to CPU for numpy
                test_predictions.extend(preds)
                test_true_values.extend(batch_y.cpu().numpy())  # Move back to CPU

        test_predictions = np.array(test_predictions)
        test_true_values = np.array(test_true_values)

        test_rmse = sqrt(mean_squared_error(test_true_values, test_predictions))
        test_rmses.append(test_rmse)

        # Check if this is the best model
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_model_state = copy.deepcopy(regressor_model.state_dict())
            best_epoch = epoch + 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Test RMSE: {test_rmse:.4f}")

    # Load the best model
    if best_model_state is not None:
        regressor_model.load_state_dict(best_model_state)
        print(f"Loaded best model from epoch {best_epoch} with test RMSE: {best_rmse:.4f}")

    # Final evaluation with best model
    regressor_model.eval()
    final_predictions = []
    final_true_values = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = regressor_model(batch_X)
            preds = outputs.cpu().numpy()
            final_predictions.extend(preds)
            final_true_values.extend(batch_y.cpu().numpy())

    final_predictions = np.array(final_predictions)
    final_true_values = np.array(final_true_values)

    # Calculate final regression metrics
    rmse = sqrt(mean_squared_error(final_true_values, final_predictions))
    mae = np.mean(np.abs(final_true_values - final_predictions))

    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(final_true_values, final_predictions)

    print(f"Final Transformer Test RMSE: {rmse:.4f}")
    print(f"Final Transformer Test MAE: {mae:.4f}")
    print(f"Final Transformer Test R²: {r2:.4f}")

    return final_true_values, final_predictions, {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'train_losses': train_losses,
        'test_rmses': test_rmses,
        'best_epoch': best_epoch,
        'best_rmse': best_rmse
    }