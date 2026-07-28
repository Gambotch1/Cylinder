# src/models/gru.py

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Optional
import pandas as pd

from viv_analysis.utils import parse_ur_label


class VIV_GRU(nn.Module):
    """
    GRU surrogate for aerodynamic lift coefficient prediction.
    
    Processes kinematics as a sequence.
    Hidden state carries phase information across timesteps.
    
    Input shape:  (batch, seq_len, n_features)
    Output shape: (batch,)  — CL at the last timestep
    """
    def __init__(
        self,
        input_size:  int = 3,     # [disp, vel, acc]
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x:  torch.Tensor,                    # (batch, seq_len, input_size)
        h0: Optional[torch.Tensor] = None,   # (num_layers, batch, hidden_size)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, hn = self.gru(x, h0)
        pred = self.head(out[:, -1, :]).squeeze(-1)   # (batch,)
        return pred, hn


class VIVSequenceDataset(Dataset):
    """
    Dataset that returns (sequence, target) pairs.

    sequence shape: (seq_len, n_features)
    target:         scalar CL value

    Unlike the ELM dataset, we do NOT flatten — the GRU needs the
    sequence structure.
    """
    def __init__(
        self,
        df:           pd.DataFrame,
        seq_len:      int,
        target_col:   str,
        input_cols:   list[str],
        release_time: dict[str, float],
        stride:       int = 1,
        use_ur_context: bool = False,
        ur_mean: float = 0.0,
        ur_std: float = 1.0,
    ):
        self.sequences:  list[np.ndarray] = []
        self.targets:    list[float]      = []
        self.case_names: list[str]        = []
        ur_std_safe = float(ur_std) if abs(float(ur_std)) > 0 else 1.0

        for case_name, case_df in df.groupby("case", sort=True):
            ordered = case_df.sort_values(["time", "step"]).reset_index(drop=True)
            signal  = ordered[input_cols].to_numpy(dtype=np.float32)

            if use_ur_context:
                ur_val = parse_ur_label(str(case_name))
                ur_scaled = (ur_val - float(ur_mean)) / ur_std_safe
                ur_col = np.full((signal.shape[0], 1), ur_scaled, dtype=np.float32)
                signal = np.hstack([signal, ur_col])

            target  = ordered[target_col].to_numpy(dtype=np.float32)
            times   = ordered["time"].to_numpy(dtype=np.float32)

            release_t   = float(release_time.get(str(case_name), -np.inf))
            release_idx = int(np.searchsorted(times, release_t, side="left"))
            start_i     = max(seq_len, release_idx + seq_len)

            if len(ordered) <= seq_len:
                continue

            for i in range(start_i, len(ordered), stride):
                self.sequences.append(signal[i - seq_len : i].copy())
                self.targets.append(float(target[i]))
                self.case_names.append(str(case_name))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple:
        x = torch.from_numpy(self.sequences[idx])   # (seq_len, n_features)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)  # scalar
        return x, y, self.case_names[idx]


def fit_scalers(
    train_df:     pd.DataFrame,
    input_cols:   list[str],
    target_col:   str,
) -> tuple[StandardScaler, StandardScaler]:
    """Fit scalers on training data only."""
    x_vals = train_df[input_cols].to_numpy(dtype=np.float32)
    y_vals = train_df[target_col].to_numpy(dtype=np.float32)
    x_scaler = StandardScaler().fit(x_vals)
    y_scaler = StandardScaler().fit(y_vals.reshape(-1, 1))
    return x_scaler, y_scaler


def apply_scalers_to_df(
    df:       pd.DataFrame,
    x_scaler: StandardScaler,
    y_scaler: Optional[StandardScaler],
    input_cols:  list[str],
    target_col:  str,
) -> pd.DataFrame:
    """Return a copy of df with input cols and target col scaled."""
    df = df.copy()
    df[input_cols]  = x_scaler.transform(df[input_cols].to_numpy(dtype=np.float32))
    if y_scaler is not None:
        df[target_col]  = y_scaler.transform(
            df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        ).ravel()
    return df
