from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElementState:
    ke_total: np.ndarray
    fe_int: np.ndarray
    strain: float


def truss1d_linear_element(
    x_i0: float,
    x_j0: float,
    u_i: float,
    u_j: float,
    young: float,
    area: float,
) -> ElementState:
    """Barra 1D lineal clásica - pequeños desplazamientos.

    Matriz de rigidez: K = (EA/L) * [1, -1; -1, 1]
    Deformación lineal: epsilon = (u_j - u_i) / L
    """
    l0 = abs(x_j0 - x_i0)
    if l0 <= 0.0:
        raise ValueError("Element with zero initial length detected")

    # Deformación lineal (pequeños desplazamientos)
    epsilon = (u_j - u_i) / l0

    # Rigidez constitutiva lineal: K = (EA/L) * [1, -1; -1, 1]
    stiffness = (young * area) / l0
    ke = stiffness * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)

    # Fuerza interna: F_int = K * u_elem = (EA/L) * [u_i - u_j, u_j - u_i]
    fe_int = stiffness * np.array([u_i - u_j, u_j - u_i], dtype=float)

    return ElementState(ke_total=ke, fe_int=fe_int, strain=epsilon)


def truss2d_linear_element(
    x_i0: np.ndarray,
    x_j0: np.ndarray,
    u_i: np.ndarray,
    u_j: np.ndarray,
    young: float,
    area: float,
) -> ElementState:
    """Barra 2D lineal - pequeños desplazamientos.

    Extiende el elemento 1D a 2D usando cosenos directores.
    Matriz 4x4 con rigidez solo en dirección axial de la barra.
    Deformación lineal: epsilon = (du_axial) / L
    """
    dx0 = x_j0 - x_i0
    l0 = float(np.linalg.norm(dx0))
    if l0 <= 0.0:
        raise ValueError("Element with zero initial length detected")

    # Cosenos directores (dirección de la barra)
    cx = dx0[0] / l0  # cos(θ)
    cy = dx0[1] / l0  # sin(θ)

    # Desplazamiento relativo axial (proyección en dirección de la barra)
    du = u_j - u_i
    du_axial = cx * du[0] + cy * du[1]

    # Deformación lineal
    epsilon = du_axial / l0

    # Rigidez constitutiva
    stiffness = (young * area) / l0

    # Matriz de rigidez global 4x4 (solo componente axial)
    # K = (EA/L) * [c²  cs  -c²  -cs ]
    #              [cs  s²  -cs  -s² ]
    #              [-c² -cs  c²   cs ]
    #              [-cs -s²  cs   s² ]
    # donde c = cos(θ), s = sin(θ)
    c2 = cx * cx
    s2 = cy * cy
    cs = cx * cy

    ke = stiffness * np.array(
        [
            [c2, cs, -c2, -cs],
            [cs, s2, -cs, -s2],
            [-c2, -cs, c2, cs],
            [-cs, -s2, cs, s2],
        ],
        dtype=float,
    )

    # Fuerza interna: F_int = K * u_elem
    u_elem = np.array([u_i[0], u_i[1], u_j[0], u_j[1]], dtype=float)
    fe_int = ke @ u_elem

    return ElementState(ke_total=ke, fe_int=fe_int, strain=epsilon)


def truss2d_element_state(
    x_i0: np.ndarray,
    x_j0: np.ndarray,
    u_i: np.ndarray,
    u_j: np.ndarray,
    young: float,
    area: float,
) -> ElementState:
    """Truss 2D con no linealidad geométrica (Green-Lagrange)."""
    dx0 = x_j0 - x_i0
    l0 = float(np.linalg.norm(dx0))
    if l0 <= 0.0:
        raise ValueError("Element with zero initial length detected")

    x_i = x_i0 + u_i
    x_j = x_j0 + u_j
    dx = x_j - x_i
    l = float(np.linalg.norm(dx))

    d = np.array([dx[0], dx[1], -dx[0], -dx[1]], dtype=float)
    d0 = np.array([dx0[0], dx0[1], -dx0[0], -dx0[1]], dtype=float)

    e_gl = (l * l - l0 * l0) / (2.0 * l0 * l0)

    ke_l = (young * area / (l0**3)) * np.outer(d0, d0)
    ke_nl = (young * area / l0) * e_gl * np.outer(d, d)
    fe_int = (young * area / l0) * e_gl * d

    return ElementState(ke_total=ke_l + ke_nl, fe_int=fe_int, strain=e_gl)
