from __future__ import annotations

import numpy as np

from fem import FEMModel, Material, SolverConfig, solve_incremental_newton


def build_geometry_fem2d_like(height: int = 20, span: int = 3) -> tuple[np.ndarray, np.ndarray]:
    x: list[float] = []
    y: list[float] = []
    t: list[list[int]] = []

    for i in range(1, height + 1):
        xtri = [0.0, 1.0]
        ytri = [float(i - 1), float(i - 1)]
        x.extend(xtri)
        y.extend(ytri)

        if i > 1:
            base = 2 * (i - 2)
            ttri = [[1, 2], [2, 4], [4, 1], [1, 3]]
            for a, b in ttri:
                t.append([a + base, b + base])

        if i == height:
            base = 2 * (i - 2)
            t.append([3 + base, 4 + base])

    tmax = max(max(edge) for edge in t)
    t_extra = [[a + tmax, b + tmax] for a, b in t]
    t.extend(t_extra)
    tmax = max(max(edge) for edge in t)

    x2 = [xi + span for xi in x]
    y2 = [yi for yi in y]
    x.extend(x2)
    y.extend(y2)

    for i in range(1, span):
        if i != 1 and i != span - 1:
            xtri = [2.0 + (i - 1), 2.0 + (i - 1)]
            ytri = [float(height - 2), float(height - 1)]
            ttri = [[1, 3], [3, 5], [5, 1], [2, 5]]
            for a, b in ttri:
                t.append([a + tmax + i - 2, b + tmax + i - 2])
        elif i == 1:
            xtri = [2.0, 2.0]
            ytri = [float(height - 2), float(height - 1)]
            ttri = [
                [2 * (height - 1) - tmax, 1],
                [1, 2],
                [2, 2 * (height - 1) - tmax],
                [2 * (height - 1) - tmax, 2],
                [2 * height - tmax, 2],
            ]
            for a, b in ttri:
                t.append([a + tmax, b + tmax])
        else:
            xtri = []
            ytri = []
            ttri = [
                [1, 2 * (height - 1) - tmax // 2 - 1],
                [2, 2 * (height - 1) - tmax // 2 + 1],
                [2 * (height - 1) - tmax // 2 + 1, 1],
            ]
            for a, b in ttri:
                t.append([a + tmax, b + tmax])

        x.extend(xtri)
        y.extend(ytri)

    nodes = np.column_stack([np.array(x, dtype=float), np.array(y, dtype=float)])
    elements = np.array(t, dtype=int) - 1
    return nodes, elements


def build_model() -> FEMModel:
    nodes, elements = build_geometry_fem2d_like(height=20, span=3)
    nnode = nodes.shape[0]
    ndof = nnode * 2

    loads = np.zeros(ndof, dtype=float)

    prescribed_dofs_1b = np.array([37, 38, 39, 40], dtype=int)
    prescribed_vals = 1e8 * np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    loads[prescribed_dofs_1b - 1] -= prescribed_vals

    fixed_dofs_1b = np.array([1, 2, 3, 4, 39, 40, 41, 42], dtype=int)
    fixed_dofs = fixed_dofs_1b - 1

    material = Material(young=2.1e11, area=1.0, density=7800.0)
    return FEMModel(
        nodes=nodes,
        elements=elements,
        material=material,
        loads=loads,
        fixed_dofs=fixed_dofs,
    )


def main() -> None:
    model = build_model()
    config = SolverConfig(n_increments=10, max_iterations=120, tolerance=1e-5)
    result = solve_incremental_newton(model, config=config)

    last = result.history[-1]
    max_u = float(np.max(np.linalg.norm(result.displacements, axis=1)))
    print(f"Converged: {result.converged}")
    print(
        "Last increment -> "
        f"iterations: {int(last['iterations'])}, residual: {last['residual']:.3e}, max_e_gl: {last['max_e_gl']:.3e}"
    )
    print(f"Max nodal displacement norm: {max_u:.6e}")


if __name__ == "__main__":
    main()
