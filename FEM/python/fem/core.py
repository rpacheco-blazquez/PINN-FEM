from __future__ import annotations

import numpy as np

from .assembly import assemble_system
from .boundary import free_and_fixed_dofs
from .model import FEMModel, SolverConfig, SolverResult


def solve_incremental_newton(model: FEMModel, config: SolverConfig | None = None) -> SolverResult:
    config = config or SolverConfig()

    u = np.zeros(model.ndof, dtype=float)
    free_dofs, fixed_dofs = free_and_fixed_dofs(model.ndof, model.fixed_dofs)
    history = []
    converged_all = True

    for iinc in range(1, config.n_increments + 1):
        load_factor = iinc / config.n_increments
        f_ext = load_factor * model.loads

        has_converged = False
        residual_norm = np.inf
        max_e_gl = 0.0
        n_iter = 0

        for ite in range(config.max_iterations):
            k_tan, f_int, max_e_gl = assemble_system(model, u)
            rhs = f_ext - f_int

            k_ff = k_tan[np.ix_(free_dofs, free_dofs)]
            rhs_f = rhs[free_dofs]

            try:
                du_f = np.linalg.solve(k_ff, rhs_f)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("Tangent stiffness became singular during solve") from exc

            du = np.zeros_like(u)
            du[free_dofs] = du_f
            du[fixed_dofs] = 0.0

            u += du
            residual_norm = np.linalg.norm(du) / max(np.linalg.norm(u), config.min_denominator)
            n_iter = ite + 1

            if residual_norm <= config.tolerance:
                has_converged = True
                break

        history.append(
            {
                "increment": float(iinc),
                "load_factor": float(load_factor),
                "iterations": float(n_iter),
                "residual": float(residual_norm),
                "max_strain": float(max_e_gl),
                "converged": float(1.0 if has_converged else 0.0),
            }
        )
        converged_all = converged_all and has_converged

    k_final, f_int_final, _ = assemble_system(model, u)
    reactions = k_final @ u - model.loads
    reactions[free_dofs] = 0.0

    if model.dimension == 1:
        displacements_out = u.reshape(-1, 1)
        reactions_out = reactions.reshape(-1, 1)
    else:
        displacements_out = u.reshape(model.nnode, model.dimension)
        reactions_out = reactions.reshape(model.nnode, model.dimension)

    return SolverResult(
        displacements=displacements_out,
        reactions=reactions_out,
        converged=converged_all,
        history=history,
    )
