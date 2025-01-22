import gin
import numpy as np
from typing import Tuple, Any


@gin.configurable
class ConstraintsNormalizer:
    """ MinMax scaler in range from -1 to 1
    for each coordinate
    """
    def __init__(
        self,
        x_coord_range: Tuple[float, float] = (-851., 851.),
        y_coord_range: Tuple[float, float] = (-851., 851.),
        z_coord_range: Tuple[float, float] = (-2386., 2386.)
    ):
        self.x_min, self.x_max = x_coord_range
        self.y_min, self.y_max = y_coord_range
        self.z_min, self.z_max = z_coord_range

    def __call__(
        self,
        hits: np.ndarray[(Any, 3), np.float32]
    ) -> np.ndarray[(Any, 3), np.float32]:
        x_norm = 2 * (hits[:, 0] - self.x_min) / (self.x_max - self.x_min) - 1
        y_norm = 2 * (hits[:, 1] - self.y_min) / (self.y_max - self.y_min) - 1
        z_norm = 2 * (hits[:, 2] - self.z_min) / (self.z_max - self.z_min) - 1
        normalized_hits = np.hstack([
            x_norm.reshape(-1, 1),
            y_norm.reshape(-1, 1),
            z_norm.reshape(-1, 1)
        ])
        return normalized_hits