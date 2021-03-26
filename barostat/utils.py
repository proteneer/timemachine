import numpy as np


def compute_volume(box: np.ndarray) -> float:
    return np.linalg.det(box)

