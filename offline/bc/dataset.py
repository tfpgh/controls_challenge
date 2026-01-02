import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from offline.config import BCConfig


class BCDataset(Dataset):
    """
    Dataset of (state, action) pairs extracted from PGTO trajectories.

    Each sample has:
        The current state (target, lataccel, roll, v_ego, a_ego)
        Past actions (20 steps)
        Past lataccels (20 steps)
        Future trajectory (50 steps of target, roll, v_ego, a_ego)
        Timestep features (segment progress and past action validity)

    Label is the PGTO action at that state
    """

    LATACCEL_BINS = torch.linspace(-5, 5, 1024)

    def __init__(
        self,
        segment_ids: list[str],
        config: BCConfig,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.pgto_dir = Path(config.pgto_data_dir)
        self.segments_dir = Path(config.segments_dir)

        self.features: torch.Tensor  # [N, 247]
        self.labels: torch.Tensor  # [N]

        self._load_data_cached(segment_ids, verbose)

    def _load_data_cached(
        self,
        segment_ids: list[str],
        verbose: bool,
    ) -> None:
        """Load data from cache if available, otherwise process and cache."""
        # Create cache key from sorted segment IDs
        cache_key = hashlib.md5("_".join(sorted(segment_ids)).encode()).hexdigest()[:12]
        cache_path = self.pgto_dir / f"bc_cache_{cache_key}.pt"

        if cache_path.exists():
            if verbose:
                print(f"Loading from cache: {cache_path}")
            cached = torch.load(cache_path, weights_only=True)
            self.features = cached["features"]
            self.labels = cached["labels"]
            if verbose:
                print(f"Loaded {len(self)} samples from cache")
        else:
            if verbose:
                print("Cache not found, processing segments...")
            self._load_data(segment_ids, verbose)

            if verbose:
                print(f"Saving cache to: {cache_path}")
            torch.save({"features": self.features, "labels": self.labels}, cache_path)

    def _load_data(
        self,
        segment_ids: list[str],
        verbose: bool,
    ) -> None:
        """
        Load and preprocess all samples into memory.
        """
        all_features = []
        all_labels = []

        iterator = (
            tqdm(segment_ids, desc="Loading segments") if verbose else segment_ids
        )

        for seg_id in iterator:
            pt_path = self.pgto_dir / f"{seg_id}.pt"
            csv_path = self.segments_dir / f"{seg_id}.csv"

            if not pt_path.exists() or not csv_path.exists():
                continue

            seg_features, seg_labels = self._process_segment(pt_path, csv_path)
            all_features.append(seg_features)
            all_labels.append(seg_labels)

        self.features = torch.cat(all_features, dim=0)
        self.labels = torch.cat(all_labels, dim=0)

        if verbose:
            print(f"Loaded {len(self)} samples from {len(segment_ids)} segments")

    def _process_segment(
        self, pt_path: Path, csv_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one segment, extracting samples from all restarts."""
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        num_restarts = data["num_restarts"]

        # Load CSV for future context
        df = pd.read_csv(csv_path)
        roll_full = torch.tensor(np.sin(df["roll"].values) * 9.81, dtype=torch.float32)
        v_ego_full = torch.tensor(df["vEgo"].values, dtype=torch.float32)
        a_ego_full = torch.tensor(df["aEgo"].values, dtype=torch.float32)

        all_features = []
        all_labels = []

        for r in range(num_restarts):
            traj = data[f"restart_{r}_trajectory"]
            features, labels = self._process_trajectory(
                traj, roll_full, v_ego_full, a_ego_full
            )
            all_features.append(features)
            all_labels.append(labels)

        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def _process_trajectory(
        self,
        traj: dict[str, torch.Tensor],
        roll_full: torch.Tensor,
        v_ego_full: torch.Tensor,
        a_ego_full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract samples from one trajectory.
        """
        # Unpack trajectory data
        history_states = traj["history_states"]  # [T, 20, 4]
        history_tokens = traj["history_tokens"]  # [T, 20]
        current_lataccel = traj["current_lataccel"]  # [T]
        targets = traj["targets"]  # [T]
        actions = traj["actions"]  # [T]

        T = len(actions)
        control_start = self.config.control_start_idx
        context_len = self.config.context_length
        future_len = self.config.future_length

        csv_offset = control_start

        features_list = []

        for t in range(T):
            csv_t = csv_offset + t

            # Current state [5]
            current_state = torch.tensor(
                [
                    targets[t].item(),
                    current_lataccel[t].item(),
                    history_states[t, -1, 1].item(),  # roll
                    history_states[t, -1, 2].item(),  # v_ego
                    history_states[t, -1, 3].item(),  # a_ego
                ]
            )

            # Past actions [20] - zeros before control, then PGTO actions
            # At trajectory step t, we've taken t actions (0 to t-1)
            # history_states[t, :, 0] contains actions from steps t-20 to t-1
            past_actions = history_states[t, :, 0].clone()  # [20]

            # Zero out actions that would be from before control started
            # If t < 20, some of these are from warmup (CSV actions, not ours)
            # To match eval: zeros for any step before we had real data
            steps_of_real_actions = t  # We've taken t real actions so far
            if steps_of_real_actions < context_len:
                # Zero out the ones that aren't real
                num_to_zero = context_len - steps_of_real_actions
                past_actions[:num_to_zero] = 0.0

            # Past lataccels [20] - decode from tokens
            past_lataccels = self.LATACCEL_BINS[history_tokens[t].long()]  # [20]

            # Future targets [50]
            future_targets = self._get_future_padded(targets, t, future_len)

            # Future roll/v_ego/a_ego [50 each] - from CSV
            future_roll = self._get_future_padded_from_full(
                roll_full, csv_t, future_len
            )
            future_v_ego = self._get_future_padded_from_full(
                v_ego_full, csv_t, future_len
            )
            future_a_ego = self._get_future_padded_from_full(
                a_ego_full, csv_t, future_len
            )

            # Timestep features [2]
            segment_progress = t / T
            action_buffer_validity = min(t, context_len) / context_len

            timestep_features = torch.tensor([segment_progress, action_buffer_validity])

            # Concatenate all features [247]
            sample_features = torch.cat(
                [
                    current_state,  # [5]     indices 0-4
                    past_actions,  # [20]    indices 5-24
                    past_lataccels,  # [20]    indices 25-44
                    future_targets,  # [50]    indices 45-94
                    future_roll,  # [50]    indices 95-144
                    future_v_ego,  # [50]    indices 145-194
                    future_a_ego,  # [50]    indices 195-244
                    timestep_features,  # [2]     indices 245-246
                ]
            )

            features_list.append(sample_features)

        features = torch.stack(features_list, dim=0)  # [T, 247]
        labels = actions.clone()  # [T]

        return features, labels

    def _get_future_padded(
        self, tensor: torch.Tensor, t: int, length: int
    ) -> torch.Tensor:
        """
        Get future values starting from t+1, padded to length.
        """
        future = tensor[t + 1 : t + 1 + length]
        if len(future) < length:
            pad_val = future[-1] if len(future) > 0 else tensor[t]
            padding = pad_val.expand(length - len(future))
            future = torch.cat([future, padding])
        return future

    def _get_future_padded_from_full(
        self, full_tensor: torch.Tensor, csv_t: int, length: int
    ) -> torch.Tensor:
        """
        Get future values from full CSV tensor, padded to length.
        """
        start = csv_t + 1
        end = start + length
        future = full_tensor[start:end]
        if len(future) < length:
            pad_val = future[-1] if len(future) > 0 else full_tensor[csv_t]
            padding = pad_val.expand(length - len(future))
            future = torch.cat([future, padding])
        return future

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def get_segment_ids(pgto_dir: Path) -> list[str]:
    """
    Get all available segment IDs from PGTO output directory.
    """
    return sorted([p.stem for p in pgto_dir.glob("*.pt")])
