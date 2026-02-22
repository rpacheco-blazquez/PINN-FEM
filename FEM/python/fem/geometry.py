from __future__ import annotations

from typing import Tuple

import numpy as np


def element_dofs(node_i: int, node_j: int) -> np.ndarray:
    return np.array([2 * node_i, 2 * node_i + 1, 2 * node_j, 2 * node_j + 1], dtype=int)


def split_element_data(nodes: np.ndarray, disp: np.ndarray, node_i: int, node_j: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_i0 = nodes[node_i]
    x_j0 = nodes[node_j]

    u_i = np.array([disp[2 * node_i], disp[2 * node_i + 1]], dtype=float)
    u_j = np.array([disp[2 * node_j], disp[2 * node_j + 1]], dtype=float)
    return x_i0, x_j0, u_i, u_j
