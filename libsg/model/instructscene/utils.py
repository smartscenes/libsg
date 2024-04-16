import os
from typing import Any, Optional

import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel


def load_checkpoints(
    model: nn.Module,
    ckpt_dir: str,
    ema_states: Optional[EMAModel] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    device=torch.device("cpu"),
) -> int:
    """Load checkpoint from the given experiment directory and return the epoch of this checkpoint."""
    if epoch is not None and epoch < 0:
        epoch = None

    model_files = [f.split(".")[0] for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pth")]

    if len(model_files) == 0:  # no checkpoints found
        print(f"No checkpoint found in {ckpt_dir}, starting from scratch\n")
        return -1

    epoch = epoch or max([int(f[6:]) for f in model_files])  # load the latest checkpoint by default
    checkpoint_path = os.path.join(ckpt_dir, f"epoch_{epoch:05d}.pth")
    if not os.path.exists(checkpoint_path):  # checkpoint file not found
        print(f"Checkpoint file {checkpoint_path} not found, starting from scratch\n")
        return -1

    print(f"Load checkpoint from {checkpoint_path}\n")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    if ema_states is not None:
        ema_states.load_state_dict(checkpoint["ema_states"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return epoch
