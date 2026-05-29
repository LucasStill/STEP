import torch.nn.functional as F
import torch

# Triplet loss function
def triplet_loss(anchor, positive, negative, margin=1.0):
    """Calculate triplet loss with cosine similarity"""
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    losses = F.relu(neg_sim - pos_sim + margin)
    return losses.mean()

# Optimized prototype loss - only for endpoints
def prototype_loss(z, is_initial, is_failure, initial_prototype, failure_prototype):
    """
    Loss to pull endpoint embeddings toward appropriate prototypes
    Only applies to initial and failure points, middle points are ignored
    """
    device = z.device

    # Convert to tensors if needed
    if not torch.is_tensor(is_initial):
        is_initial = torch.tensor(is_initial, device=device).float()
    if not torch.is_tensor(is_failure):
        is_failure = torch.tensor(is_failure, device=device).float()

    prototype_loss = torch.tensor(0.0).to(device)
    n_prototype_samples = 0

    # Initial prototype loss
    if is_initial.sum() > 0:
        initial_embeddings = z[is_initial.bool()]
        initial_target = initial_prototype.expand_as(initial_embeddings)
        prototype_loss += F.mse_loss(initial_embeddings, initial_target)
        n_prototype_samples += 1

    # Failure prototype loss
    if is_failure.sum() > 0:
        failure_embeddings = z[is_failure.bool()]
        failure_target = failure_prototype.expand_as(failure_embeddings)
        prototype_loss += F.mse_loss(failure_embeddings, failure_target)
        n_prototype_samples += 1

    # Average if we have any prototype samples
    if n_prototype_samples > 0:
        prototype_loss /= n_prototype_samples

    #print(f'prototype: {prototype_loss.shape}')
    return prototype_loss

# Optimized combined loss
def combined_loss(anchor, positive, negative, reconstruction, target, z_anchor,
                 is_initial, is_failure, model, alpha=0.5, beta=0.2, margin=1.0):
    """Combine reconstruction loss with triplet loss and prototype loss"""
    recon_loss = F.mse_loss(reconstruction, target)
    trip_loss = triplet_loss(z_anchor, positive, negative, margin)
    proto_loss = prototype_loss(z_anchor, is_initial, is_failure,
                               model.initial_prototype, model.failure_prototype)

    total_loss = recon_loss + alpha * trip_loss + beta * proto_loss
    return total_loss, recon_loss, trip_loss, proto_loss
