from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

def extract_latent_representations(model, X_data, device, batch_size=512):
    """Extract latent representations in batches to avoid memory issues"""
    model.eval()
    latent_vectors = []

    # Create DataLoader for efficient batch processing
    dataset = TensorDataset(torch.FloatTensor(X_data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch[0].to(device)
            # Use the encode method which returns just the latent vectors
            z_batch = model.encode(x_batch)
            latent_vectors.append(z_batch.cpu().numpy())

    return np.vstack(latent_vectors)