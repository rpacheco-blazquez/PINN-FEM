from __future__ import annotations

from typing import Tuple

import numpy as np


def free_and_fixed_dofs(ndof: int, fixed_dofs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fixed = np.unique(np.asarray(fixed_dofs, dtype=int).reshape(-1))
    mask = np.ones(ndof, dtype=bool)
    mask[fixed] = False
    free = np.flatnonzero(mask)
    return free, fixed
