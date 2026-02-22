from __future__ import annotations

from typing import Tuple

import numpy as np

from .element import (
    truss1d_linear_element,
    truss2d_linear_element,
    truss2d_element_state,
)
from .geometry import element_dofs, split_element_data
from .model import FEMModel


def assemble_system(
    model: FEMModel, disp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    k_global = np.zeros((model.ndof, model.ndof), dtype=float)
    f_int = np.zeros(model.ndof, dtype=float)
    max_abs_strain = 0.0

    if model.dimension == 1:
        # 1D truss linear
        for node_i, node_j in model.elements:
            dof_i = node_i
            dof_j = node_j
            dofs = np.array([dof_i, dof_j], dtype=int)

            x_i0 = model.nodes[node_i]
            x_j0 = model.nodes[node_j]
            u_i = disp[dof_i]
            u_j = disp[dof_j]

            # Evaluar propiedades materiales en el centro del elemento
            x_center = (x_i0 + x_j0) / 2.0
            young_val = model.material.young.value(x_center)
            area_val = model.material.area.value(x_center)

            elm = truss1d_linear_element(
                x_i0=x_i0,
                x_j0=x_j0,
                u_i=u_i,
                u_j=u_j,
                young=young_val,
                area=area_val,
            )
            k_global[np.ix_(dofs, dofs)] += elm.ke_total
            f_int[dofs] += elm.fe_int
            max_abs_strain = max(max_abs_strain, abs(elm.strain))

    elif model.dimension == 2:
        # 2D truss LINEAR (small displacements)
        for node_i, node_j in model.elements:
            dofs = element_dofs(node_i, node_j)
            x_i0, x_j0, u_i, u_j = split_element_data(model.nodes, disp, node_i, node_j)

            # Evaluar propiedades materiales en el centro del elemento
            x_center = (x_i0 + x_j0) / 2.0
            young_val = model.material.young.value(x_center)
            area_val = model.material.area.value(x_center)

            elm = truss2d_linear_element(
                x_i0=x_i0,
                x_j0=x_j0,
                u_i=u_i,
                u_j=u_j,
                young=young_val,
                area=area_val,
            )
            k_global[np.ix_(dofs, dofs)] += elm.ke_total
            f_int[dofs] += elm.fe_int
            max_abs_strain = max(max_abs_strain, abs(elm.strain))

    return k_global, f_int, max_abs_strain
