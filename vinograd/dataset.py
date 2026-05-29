import random

from torch.utils.data import Dataset
import numpy as np

class TripletWindowDataset(Dataset):
    """Custom dataset for triplet sampling with prototype support"""
    def __init__(self, windows, rul_labels, unit_ids, next_windows=None, margin=10, health_margin=20,
                 n_initial_prototypes=8, n_failure_prototypes=8, step_size=1):
        """
        Args:
            windows: Window features (n_samples, n_features)
            rul_labels: RUL values for each window
            unit_ids: Unit identifiers for each window
            margin: Minimum RUL difference for negative samples
            health_margin: Maximum RUL difference for positive samples
            n_initial_prototypes: Number of initial timesteps to use as prototypes
            n_failure_prototypes: Number of final timesteps to use as prototypes
        """
        self.windows = windows
        self.rul_labels = rul_labels
        self.unit_ids = unit_ids
        self.next_windows = next_windows  # For forecasting
        self.margin = margin
        self.health_margin = health_margin
        self.n_initial_prototypes = n_initial_prototypes
        self.n_failure_prototypes = n_failure_prototypes
        self.step_size = step_size

        # Identify prototype candidates based on trajectory position
        self.initial_prototype_indices, self.failure_prototype_indices = self._get_prototype_indices()

    def _get_prototype_indices(self):
        """Get indices for initial and failure prototypes based on trajectory position"""
        initial_indices = []
        failure_indices = []

        unique_units = np.unique(self.unit_ids)

        for unit_id in unique_units:
            # Get all indices for this unit
            unit_indices = np.where(self.unit_ids == unit_id)[0]

            if len(unit_indices) == 0:
                continue

            # Sort by RUL (descending) to get chronological order
            unit_ruls = self.rul_labels[unit_indices]
            sorted_indices = unit_indices[np.argsort(unit_ruls)[::-1]]  # High RUL first

            # Take first n_initial_prototypes as initial prototypes
            n_initial = min(self.n_initial_prototypes, len(sorted_indices))
            initial_indices.extend(sorted_indices[:n_initial])

            # Take last n_failure_prototypes as failure prototypes
            n_failure = min(self.n_failure_prototypes, len(sorted_indices))
            failure_indices.extend(sorted_indices[-n_failure:])

        return initial_indices, failure_indices

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Return triplet (anchor, positive, negative)
        doesn't sample with bins
        """
        anchor = self.windows[idx]
        anchor_rul = self.rul_labels[idx]
        anchor_unit = self.unit_ids[idx]

        health_margin = self.health_margin
        margin = self.margin

        # Get positive candidates (same unit, within health_margin of anchor)
        positive_min = max(idx - health_margin, 0)
        positive_max = min(idx + health_margin, len(self.windows))

        # Create list of positive candidates excluding the anchor itself
        positive_candidates = [
            i for i in range(positive_min, positive_max)
            if i != idx and self.unit_ids[i] == anchor_unit
        ]

        # If no valid positive candidates found (shouldn't happen often), use nearest
        if not positive_candidates:
            # Find the closest valid index that's not the anchor
            for offset in range(1, max(idx, len(self.windows) - idx)):
                candidates = []
                if idx + offset < len(self.windows) and self.unit_ids[idx + offset] == anchor_unit:
                    candidates.append(idx + offset)
                if idx - offset >= 0 and self.unit_ids[idx - offset] == anchor_unit:
                    candidates.append(idx - offset)
                if candidates:
                    positive_candidates = candidates
                    break

        pos_idx = random.choice(positive_candidates)

        negative_candidates = [
            i for i in range(len(self.windows))
            if self.unit_ids[i] == anchor_unit and
               abs(i - idx) >= 2 * health_margin and
               abs(i - pos_idx) >= 2 * health_margin and
               i not in positive_candidates and i != idx
        ]

        # If still no negative candidates (edge case), use any non-positive
        if not negative_candidates:
            negative_candidates = [
                i for i in range(len(self.windows))
                if i != idx and i not in positive_candidates
            ]

        neg_idx = random.choice(negative_candidates)

        positive = self.windows[pos_idx]
        negative = self.windows[neg_idx]

        # Return prototype flags
        is_initial = 1 if idx in self.initial_prototype_indices else 0
        is_failure = 1 if idx in self.failure_prototype_indices else 0

        # Return forecasting target if available
        if self.next_windows is not None:
            next_state = self.next_windows[idx]
            return anchor, positive, negative, anchor_rul, is_initial, is_failure, next_state
        else:
            return anchor, positive, negative, anchor_rul, is_initial, is_failure
