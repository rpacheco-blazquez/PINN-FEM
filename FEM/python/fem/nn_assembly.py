"""Ensamblaje con PyTorch para mantener gradientes respecto a parámetros NN.

A diferencia de assembly.py (NumPy), este módulo usa torch.Tensor
para preservar el computational graph y calcular ∂K/∂θ.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import numpy as np

from .model import FEMModel


def truss1d_linear_element_torch(
    x_i0: float,
    x_j0: float,
    u_i: torch.Tensor,
    u_j: torch.Tensor,
    young: torch.Tensor,
    area: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Elemento 1D lineal con PyTorch.

    Returns:
        ke_total: (2, 2) stiffness matrix
        fe_int: (2,) internal force vector
    """
    l0 = abs(x_j0 - x_i0)
    if l0 <= 0.0:
        raise ValueError("Element with zero initial length")

    # K = (EA/L) * [1, -1; -1, 1]
    stiffness = (young * area) / l0

    # Crear matriz de rigidez (patrón constante, pero escalada por propiedades)
    pattern = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)
    ke = stiffness * pattern

    # F_int = K * u_elem
    u_elem = torch.stack([u_i, u_j])
    fe_int = ke @ u_elem

    return ke, fe_int


def truss2d_linear_element_torch(
    x_i0: np.ndarray,
    x_j0: np.ndarray,
    u_i: torch.Tensor,
    u_j: torch.Tensor,
    young: torch.Tensor,
    area: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Elemento 2D lineal con PyTorch - pequeños desplazamientos.

    Returns:
        ke_total: (4, 4) stiffness matrix
        fe_int: (4,) internal force vector
    """
    dx0_np = x_j0 - x_i0
    l0_val = float(np.linalg.norm(dx0_np))
    if l0_val <= 0.0:
        raise ValueError("Element with zero initial length")

    # Cosenos directores (constantes, geometría inicial)
    cx = float(dx0_np[0] / l0_val)
    cy = float(dx0_np[1] / l0_val)

    # Rigidez constitutiva
    stiffness = (young * area) / l0_val
    # Ensure stiffness is a scalar (squeeze any extra dimensions)
    if isinstance(stiffness, torch.Tensor):
        stiffness = stiffness.squeeze()

    # Matriz de rigidez 4x4 (solo componente axial)
    c2 = cx * cx
    s2 = cy * cy
    cs = cx * cy

    # Crear patrón como tensor de torch para mantener compatibilidad con autograd
    ke_pattern = torch.tensor(
        [
            [c2, cs, -c2, -cs],
            [cs, s2, -cs, -s2],
            [-c2, -cs, c2, cs],
            [-cs, -s2, cs, s2],
        ],
        dtype=torch.float32,
        requires_grad=False,
    )  # Geometría no depende de u

    ke = stiffness * ke_pattern

    # Fuerza interna: F_int = K * u_elem
    u_elem = torch.cat([u_i, u_j])  # [u_i_x, u_i_y, u_j_x, u_j_y]
    fe_int = ke @ u_elem

    return ke, fe_int


def assemble_system_torch(
    model: FEMModel,
    disp: torch.Tensor,
    load_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensamblaje global usando PyTorch tensors.

    Args:
        model: FEM model con material properties (pueden ser NNProperty)
        disp: (ndof,) torch.Tensor - desplazamientos actuales
        load_factor: Factor de carga actual (0.0 a 1.0) para NNs no lineales

    Returns:
        k_global: (ndof, ndof) torch.Tensor - matriz de rigidez global
        f_int: (ndof,) torch.Tensor - fuerzas internas globales

    NOTA: Si model.material contiene NNProperty con parámetros entrenables,
    k_global y f_int mantendrán gradientes respecto a esos parámetros.
    Las NNs pueden usar load_factor para modelar comportamiento no lineal.
    """
    ndof = model.ndof
    k_global = torch.zeros((ndof, ndof), dtype=torch.float32)
    f_int = torch.zeros(ndof, dtype=torch.float32)

    if model.dimension == 1:
        # Elementos 1D lineales
        for node_i, node_j in model.elements:
            dof_i = node_i
            dof_j = node_j

            x_i0 = model.nodes[node_i]
            x_j0 = model.nodes[node_j]
            u_i = disp[dof_i]
            u_j = disp[dof_j]

            # Evaluar propiedades materiales en el centro del elemento
            x_center = (x_i0 + x_j0) / 2.0

            # Preparar inputs para NNs (incluye load_factor para no linealidad)
            if isinstance(x_center, (int, float)):
                # 1D: (x, load_factor)
                inputs = {"x": float(x_center), "load_factor": load_factor}
            else:
                # 2D: (x, y, load_factor)
                inputs = {
                    "x": float(x_center[0]),
                    "y": float(x_center[1]) if len(x_center) > 1 else 0.0,
                    "load_factor": load_factor,
                }

            # value() retorna escalar Python o torch.Tensor dependiendo del Property
            young_val = model.material.young.value(inputs)
            area_val = model.material.area.value(inputs)

            # Asegurar que son tensores
            if not isinstance(young_val, torch.Tensor):
                young_val = torch.tensor(young_val, dtype=torch.float32)
            if not isinstance(area_val, torch.Tensor):
                area_val = torch.tensor(area_val, dtype=torch.float32)

            ke, fe = truss1d_linear_element_torch(
                x_i0=x_i0,
                x_j0=x_j0,
                u_i=u_i,
                u_j=u_j,
                young=young_val,
                area=area_val,
            )

            # Ensamblaje global
            dofs = [dof_i, dof_j]
            for i_local, i_global in enumerate(dofs):
                f_int[i_global] += fe[i_local]
                for j_local, j_global in enumerate(dofs):
                    k_global[i_global, j_global] += ke[i_local, j_local]

    elif model.dimension == 2:
        # Elementos 2D lineales
        for node_i, node_j in model.elements:
            # DOFs: [ux_i, uy_i, ux_j, uy_j]
            dof_ux_i = 2 * node_i
            dof_uy_i = 2 * node_i + 1
            dof_ux_j = 2 * node_j
            dof_uy_j = 2 * node_j + 1
            dofs = [dof_ux_i, dof_uy_i, dof_ux_j, dof_uy_j]

            # Coordenadas y desplazamientos
            x_i0 = model.nodes[node_i]
            x_j0 = model.nodes[node_j]
            u_i = disp[[dof_ux_i, dof_uy_i]]  # Vista, mantiene autograd
            u_j = disp[[dof_ux_j, dof_uy_j]]  # Vista, mantiene autograd

            # Evaluar propiedades materiales en el centro del elemento
            x_center = (x_i0 + x_j0) / 2.0

            # Preparar inputs para NNs (incluye load_factor para no linealidad)
            inputs = {
                "x": float(x_center[0]),
                "y": float(x_center[1]) if len(x_center) > 1 else 0.0,
                "load_factor": load_factor,
            }

            young_val = model.material.young.value(inputs)
            area_val = model.material.area.value(inputs)

            # Asegurar que son tensores
            if not isinstance(young_val, torch.Tensor):
                young_val = torch.tensor(young_val, dtype=torch.float32)
            if not isinstance(area_val, torch.Tensor):
                area_val = torch.tensor(area_val, dtype=torch.float32)

            ke, fe = truss2d_linear_element_torch(
                x_i0=x_i0,
                x_j0=x_j0,
                u_i=u_i,
                u_j=u_j,
                young=young_val,
                area=area_val,
            )

            # Ensamblaje global
            for i_local, i_global in enumerate(dofs):
                f_int[i_global] += fe[i_local]
                for j_local, j_global in enumerate(dofs):
                    k_global[i_global, j_global] += ke[i_local, j_local]

    return k_global, f_int


def compute_residual_and_jacobian(
    model: FEMModel,
    disp: torch.Tensor,
    f_ext: torch.Tensor,
    free_dofs: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcula residual R = K*u - F y jacobiano J = ∂R/∂u = K.

    Args:
        model: FEM model
        disp: (ndof,) desplazamientos actuales
        f_ext: (ndof,) fuerzas externas
        free_dofs: índices de DOFs libres

    Returns:
        residual: (n_free,) residual en DOFs libres
        jacobian: (n_free, n_free) jacobiano ∂R/∂u en DOFs libres
    """
    k_global, f_int = assemble_system_torch(model, disp)

    # Residual completo: R = f_int - f_ext  (queremos que K*u - F = 0, equivalente a f_int - f_ext = 0)
    residual_full = f_int - f_ext

    # Extraer solo DOFs libres
    residual = residual_full[free_dofs]
    jacobian = k_global[free_dofs, :][:, free_dofs]

    return residual, jacobian
