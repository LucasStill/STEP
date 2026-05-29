import torch.nn as nn
import torch

from losses.losses import combined_loss


class SimpleTransformerTriplet(nn.Module):
    def __init__(self, input_dim, output_dim=None, latent_dim=2, d_model=64, nhead=4, num_layers=2, dropout=0.3, proto_size=None):
        super().__init__()
        self.latent_dim = latent_dim

        if output_dim is None:
            output_dim = input_dim

        # Fixed prototype vectors with configurable size (defaults to latent_dim)
        if proto_size is None:
            proto_size = latent_dim
            
        # Ensure proto_size matches latent_dim for compatibility
        proto_size = latent_dim
        
        # Create prototype vectors
        initial_proto = [1.0] + [0.0] * (proto_size - 1)
        failure_proto = [0.0, 1.0] + [0.0] * (proto_size - 2)
        
        self.register_buffer('initial_prototype', torch.tensor(initial_proto))
        self.register_buffer('failure_prototype', torch.tensor(failure_proto))

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding (simplified)
        self.pos_encoding = SimplePositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Smaller FFN
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global pooling (simpler)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Latent projection
        self.latent_projection = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, latent_dim)
        )

        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.shape[0]

        # Project input
        x_proj = self.input_projection(x.unsqueeze(1))  # (batch_size, 1, d_model)

        # Add positional encoding
        x_encoded = self.pos_encoding(x_proj)

        # Transformer
        transformer_out = self.transformer(x_encoded)  # (batch_size, 1, d_model)

        # Pooling
        pooled = self.pool(transformer_out.transpose(1, 2)).squeeze(-1)  # (batch_size, d_model)

        # Latent representation
        z = self.latent_projection(pooled)  # (batch_size, latent_dim)

        # Reconstruction
        reconstructed = self.decoder(z)  # (batch_size, input_dim)

        return reconstructed, z

    def encode(self, x):
        with torch.no_grad():
            batch_size = x.shape[0]
            x_proj = self.input_projection(x.unsqueeze(1))
            x_encoded = self.pos_encoding(x_proj)
            transformer_out = self.transformer(x_encoded)
            pooled = self.pool(transformer_out.transpose(1, 2)).squeeze(-1)
            return self.latent_projection(pooled)


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


# Global Attention Pooling
class GlobalAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        weighted = torch.sum(x * attention_weights, dim=1)  # (batch_size, d_model)
        return weighted


# Modified training loop with optimizations
def train_epoch(model, dataloader, optimizer, device, alpha=0.5, beta=0.2, margin=1.0):
    """Optimized training epoch"""
    model.train()
    total_loss = total_recon_loss = total_trip_loss = total_proto_loss = 0

    for anchor, positive, negative, anchor_rul, is_initial, is_failure in dataloader:
        anchor = anchor.float().to(device)
        positive = positive.float().to(device)
        negative = negative.float().to(device)


        # With:
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
            z_anchor, z_positive, z_negative, recon_anchor, anchor,
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

# Evaluation function
@torch.no_grad()
def evaluate_model(model, dataloader, device, alpha=0.5, beta=0.2, margin=1.0):
    """Optimized evaluation"""
    model.eval()
    total_loss = total_recon_loss = total_trip_loss = total_proto_loss = 0

    for anchor, positive, negative, anchor_rul, is_initial, is_failure in dataloader:
        anchor = anchor.float().to(device)
        positive = positive.float().to(device)
        negative = negative.float().to(device)


        # With:
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