import numpy as np
import torch


def square_room_mask(room_dim=64, floor_perc=0.8) -> torch.Tensor:
    """Generate square room mask, i.e. the overall shape of the room in which to construct the layout.

    :param room_dim: dimension of room mask, defaults to 64
    :param floor_perc: percent of mask which corresponds to floor. The padding is rounded to the nearest int,
    i.e. round((1 - floor_perc) * room_dim). Defaults to 0.8
    :return: float tensor corresponding to room mask of 0s and 1s
    """
    pad = round((1 - floor_perc) * room_dim)
    room_mask = torch.zeros(size=(1, 1, 64, 64), dtype=torch.float32)
    room_mask[0, 0, pad : (room_dim - pad), pad : (room_dim - pad)] = 1.0
    return room_mask


def descale(x, minimum: np.ndarray | torch.Tensor | float, maximum: np.ndarray | torch.Tensor | float) -> np.ndarray | torch.Tensor | float:
    """Apply scaling to convert x from range (-1, 1) to (minimum, maximum)"""
    x = (x + 1) / 2
    x = x * (maximum - minimum) + minimum
    return x
