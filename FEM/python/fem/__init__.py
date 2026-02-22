from .model import FEMModel, Material, SolverConfig, SolverResult
from .core import solve_incremental_newton
from .properties import Property, ScalarProperty, NNProperty, to_property
from .nn_solver import solve_pinn_newton_raphson, PINNSolverConfig, PINNSolverResult
from .nn_solver_gd import solve_pinn_gradient_descent, PINNGradientDescentConfig, PINNGradientDescentResult
from .nn_assembly import assemble_system_torch, compute_residual_and_jacobian

__all__ = [
    "FEMModel",
    "Material",
    "SolverConfig",
    "SolverResult",
    "solve_incremental_newton",
    "Property",
    "ScalarProperty",
    "NNProperty",
    "to_property",
    # PINN solvers
    "solve_pinn_newton_raphson",
    "PINNSolverConfig",
    "PINNSolverResult",
    "solve_pinn_gradient_descent",
    "PINNGradientDescentConfig",
    "PINNGradientDescentResult",
    "assemble_system_torch",
    "compute_residual_and_jacobian",
]
