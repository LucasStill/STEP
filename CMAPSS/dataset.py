import numpy as np
import random
from torch.utils.data import Dataset

class TripletWindowDataset(Dataset):
    """Custom dataset for triplet sampling with prototype support"""
    def __init__(self, windows, rul_labels, unit_ids, margin=10, health_margin=20,
                 n_initial_prototypes=8, n_failure_prototypes=8):
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
        self.margin = margin
        self.health_margin = health_margin
        self.n_initial_prototypes = n_initial_prototypes
        self.n_failure_prototypes = n_failure_prototypes

        # Create health state groups
        self.health_groups = self._create_health_groups()

        # Identify prototype candidates based on trajectory position
        self.initial_prototype_indices, self.failure_prototype_indices = self._get_prototype_indices()

    def _create_health_groups(self):
        """Group indices by health state"""
        health_groups = {
            'healthy': [],      # RUL > 125
            'early_degradation': [],  # 125 >= RUL > 75
            'mid_degradation': [],    # 75 >= RUL > 30
            'late_degradation': []    # RUL <= 30
        }

        for i, rul in enumerate(self.rul_labels):
            if rul > 125:
                health_groups['healthy'].append(i)
            elif rul > 75:
                health_groups['early_degradation'].append(i)
            elif rul > 30:
                health_groups['mid_degradation'].append(i)
            else:
                health_groups['late_degradation'].append(i)

        return health_groups

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
        """Return triplet (anchor, positive, negative)"""
        anchor = self.windows[idx]
        anchor_rul = self.rul_labels[idx]
        anchor_unit = self.unit_ids[idx]

        # Get anchor health state
        anchor_health = self._get_health_state(anchor_rul)

        # Find positive sample (similar degradation state, same unit)
        positive_candidates = [
            i for i in self.health_groups[anchor_health]
            if self.unit_ids[i] == anchor_unit
            and abs(self.rul_labels[i] - anchor_rul) < self.health_margin
            and i != idx
        ]

        if positive_candidates:
            pos_idx = random.choice(positive_candidates)
            positive = self.windows[pos_idx]
        else:
            same_unit_candidates = [
                i for i in range(len(self.windows))
                if self.unit_ids[i] == anchor_unit
                and abs(self.rul_labels[i] - anchor_rul) < self.health_margin * 2
                and i != idx
            ]
            pos_idx = random.choice(same_unit_candidates) if same_unit_candidates else idx
            positive = self.windows[pos_idx]

        # Find negative sample (different degradation stage)
        different_health_states = [
            state for state in self.health_groups.keys()
            if state != anchor_health
        ]

        if different_health_states:
            target_health = random.choice(different_health_states)
            negative_candidates = self.health_groups[target_health]

            if negative_candidates:
                neg_idx = random.choice(negative_candidates)
                negative = self.windows[neg_idx]
            else:
                negative_candidates = [
                    i for i in range(len(self.windows))
                    if abs(self.rul_labels[i] - anchor_rul) > self.margin
                    and i != idx
                ]
                neg_idx = random.choice(negative_candidates) if negative_candidates else idx
                negative = self.windows[neg_idx]
        else:
            negative = anchor

        # Return prototype flags
        is_initial = 1 if idx in self.initial_prototype_indices else 0
        is_failure = 1 if idx in self.failure_prototype_indices else 0

        return anchor, positive, negative, anchor_rul, is_initial, is_failure

    def _get_health_state(self, rul):
        """Map RUL to health state"""
        if rul > 125:
            return 'healthy'
        elif rul > 75:
            return 'early_degradation'
        elif rul > 30:
            return 'mid_degradation'
        else:
            return 'late_degradation'

def create_windows(data, window_size=30, step=1, threshold=125, dataset_test_RUL=None):
    """Create windows with unit IDs and RUL labels"""
    windows = []
    rul_labels = []
    unit_ids = []

    unique_units = np.unique(data[:, 0])
    for unit_id in unique_units:
        unit_mask = data[:, 0] == unit_id
        unit_data = data[unit_mask]
        n_samples = len(unit_data)

        max_cycle = np.max(unit_data[:, 1])
        unit_rul = max_cycle - unit_data[:, 1]

        for i in range(0, n_samples - window_size + 1, step):
            window = unit_data[i:i+window_size, 2:].flatten()
            target_rul = unit_rul[i+window_size-1]

            if dataset_test_RUL is not None:
                target_rul += dataset_test_RUL[int(unit_id)-1]

            if threshold < 0 or target_rul > threshold:
                windows.append(window)
                rul_labels.append(target_rul)
                unit_ids.append(unit_id)

    return np.array(windows), np.array(rul_labels), np.array(unit_ids)

# for plots
def process_dataset(data, unit_id=None):
    if unit_id is not None:
        data = data[data[:, 0] == unit_id]
    # Return [unit_id, cycle, sensors] WITHOUT RUL
    return np.hstack((
        data[:, 0].reshape(-1, 1),   # Unit number
        data[:, 1].reshape(-1, 1),   # Cycles
        data[:, 5:]                  # Sensors
    ))