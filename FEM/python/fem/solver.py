"""Unified solver module for FEM and PINN problems.

Contains all optimization methods:
- Gradient Descent (GD): Works with or without NN parameters
- Newton-Raphson (NR): Classical FEM solver for R=0
- Hybrid NR-GD: Newton for u, Gradient Descent for θ (TODO)
- Full NR: Full Newton-Raphson with Hessian (TODO)

The solvers are agnostic to whether material properties are fixed (Scalar)
or trainable (NNProperty). When no NN parameters exist, θ=∅ and the problem
reduces to classical FEM.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .model import FEMModel
from .boundary import free_and_fixed_dofs
from .assembly import assemble_system
from .nn_assembly import assemble_system_torch


# ============================================================================
# Configuration and Result classes
# ============================================================================


@dataclass
class SolverConfig:
    """Unified configuration for all solvers."""

    # Common parameters
    max_iterations: int = 1000
    tolerance: float = 1e-6
    print_every: int = 10

    # Universal incremental loading (all solvers)
    n_increments: int = 10
    load_factor_initial: float = 0.0
    load_factor_final: float = 1.0
    min_denominator: float = 1e-10

    # Gradient Descent specific
    learning_rate_u: float = 1e-7  # Learning rate for displacements
    learning_rate_theta: float = 1e-4  # Learning rate for NN parameters

    # Loss function weights
    alpha_physics: float = 1.0  # Weight of physics residual
    alpha_data: float = 100.0  # Weight of data fitting

    # Solver method selection
    method: str = "auto"  # "auto", "gd", "nr", "hybrid", "full-nr"

    # Preconditioning options
    preconditioning: bool = False  # Enable GD preconditioning before main solve


@dataclass
class SolverResult:
    """Unified result from any solver."""

    displacements: np.ndarray
    reactions: np.ndarray
    converged: bool
    history: List[Dict[str, float]] = field(default_factory=list)

    # Optional: NN parameters (only if using NNProperty)
    nn_parameters: Optional[Dict[str, np.ndarray]] = None


# ============================================================================
# Gradient Descent Solver (works with or without NN)
# ============================================================================


def solve_gd(
    model: FEMModel,
    config: Optional[SolverConfig] = None,
    measured_disp: Optional[np.ndarray] = None,
    measured_dofs: Optional[List[int]] = None,
    target_load_factor: float = 1.0,
    u_initial: Optional[torch.Tensor] = None,
    skip_preconditioning: bool = False,  # Internal flag to skip preconditioning during hybrid solver
) -> SolverResult:
    """Gradient Descent solver for FEM/PINN problems.

    Minimizes the loss function:
        L(u, θ) = α_physics * ||R(u,θ)||² + α_data * ||u_measured - u||²

    Where:
        R(u,θ) = F_int(u,θ) - F_ext  (equilibrium residual)
        θ = NN parameters (empty if no NNProperty in material)

    Args:
        model: FEM model with material properties (Scalar or NNProperty)
        config: Solver configuration
        measured_disp: (n_measurements,) measured displacements (optional)
        measured_dofs: (n_measurements,) DOF indices for measurements (optional)
        target_load_factor: Target load factor for this solve (incremental loading)

    Returns:
        SolverResult with displacements, reactions, and convergence info
    """
    config = config or SolverConfig()

    # Handle preconditioning phase
    if config.preconditioning and not skip_preconditioning:
        print("GD Preconditioning phase...")
        precon_config = copy.deepcopy(config)
        precon_config.max_iterations = min(
            300, config.max_iterations // 3
        )  # Limited iterations
        precon_config.tolerance = max(1e-4, config.tolerance * 10)  # Relaxed tolerance
        precon_config.preconditioning = False  # Avoid infinite recursion

        try:
            precon_result = solve_gd(
                model,
                precon_config,
                measured_disp,
                measured_dofs,
                target_load_factor,
                u_initial,
                skip_preconditioning=True,
            )
            print(
                f"  Preconditioning: {precon_result.history[-1]['iteration']} iterations"
            )

            # If preconditioning already converged to tight tolerance, return it
            if (
                precon_result.converged
                and precon_result.history[-1].get("residual_norm", 1.0)
                < config.tolerance
            ):
                print(f"  Preconditioning achieved final convergence")
                return precon_result

            # Use preconditioning result as warm start for main phase
            u_initial = torch.tensor(
                precon_result.displacements.flatten(), dtype=torch.float32
            )
            print("Main GD phase (tight tolerance)...")

            # Adjust iterations for main phase
            main_config = copy.deepcopy(config)
            main_config.max_iterations = (
                config.max_iterations - precon_config.max_iterations
            )
            main_config.preconditioning = False

            main_result = solve_gd(
                model,
                main_config,
                measured_disp,
                measured_dofs,
                target_load_factor,
                u_initial,
                skip_preconditioning=True,
            )

            # Merge histories with proper iteration numbering
            precon_iterations = (
                precon_result.history[-1].get("iteration", 0)
                if precon_result.history
                else 0
            )
            unified_history = []

            if precon_result.history:
                unified_history.extend(precon_result.history)

            if main_result.history:
                for entry in main_result.history:
                    new_entry = entry.copy()
                    new_entry["iteration"] = (
                        entry.get("iteration", 0) + precon_iterations
                    )
                    unified_history.append(new_entry)

            main_result.history = unified_history
            total_iterations = (
                main_result.history[-1].get("iteration", 0)
                if main_result.history
                else 0
            )
            print(f"  Total GD (with preconditioning): {total_iterations} iterations")
            return main_result

        except Exception as e:
            print(f"  Preconditioning failed: {e}, proceeding with standard GD")

    # Get trainable NN parameters (empty list if no NNProperty)
    has_nn = model.material.has_trainable_params()
    theta_list = model.material.get_all_torch_params() if has_nn else []

    # Initialize displacements (with gradients) - use warm start if available
    if u_initial is not None:
        # Create fresh tensor with proper gradients from numpy values
        u_init_values = (
            u_initial.detach().numpy()
            if isinstance(u_initial, torch.Tensor)
            else u_initial
        )
        u = torch.tensor(u_init_values, dtype=torch.float32, requires_grad=True)
        print(f"  🔥 Using warm start from previous increment")
    else:
        u = torch.zeros(model.ndof, dtype=torch.float32, requires_grad=True)
        print(f"  ❄️  Cold start from zeros")

    f_ext_torch = torch.tensor(model.loads, dtype=torch.float32)

    # DOFs libres y fijos
    free_dofs, fixed_dofs = free_and_fixed_dofs(model.ndof, model.fixed_dofs)
    free_dofs_torch = torch.tensor(free_dofs, dtype=torch.long)
    fixed_dofs_torch = torch.tensor(fixed_dofs, dtype=torch.long)

    # Measurements (if provided)
    has_measurements = measured_disp is not None and measured_dofs is not None
    if has_measurements:
        measured_disp_torch = torch.tensor(measured_disp, dtype=torch.float32)
        measured_dofs_torch = torch.tensor(measured_dofs, dtype=torch.long)
        if config.alpha_data == 0.0:
            print("⚠️  Warning: measured_dofs provided but alpha_data=0.0")

    # Setup optimizers
    optimizer_u = torch.optim.Adam([u], lr=config.learning_rate_u)
    if theta_list:
        optimizer_theta = torch.optim.Adam(theta_list, lr=config.learning_rate_theta)
    else:
        optimizer_theta = None

    history = []
    converged = False
    best_loss = float("inf")

    # Header
    header = f"{'Iter':>6} | {'Loss Total':>12} | {'Loss Physics':>12} | {'||R||':>12} | {'Loss Data':>12} | {'||u||':>10}"
    if has_nn:
        header += f" | {'NN Params':>10}"
    print(header)
    print("-" * (82 + (12 if has_nn else 0)))

    # Optimization loop
    for iteration in range(config.max_iterations):
        # Zero gradients
        optimizer_u.zero_grad()
        if optimizer_theta:
            optimizer_theta.zero_grad()

        # ========================================
        # Assemble system and compute residuals
        # ========================================

        k_global, f_int = assemble_system_torch(
            model, u, load_factor=target_load_factor
        )

        # Physics loss: Minimize ||R||² where R = f_int - f_ext
        residual_physics = (
            f_int[free_dofs_torch] - target_load_factor * f_ext_torch[free_dofs_torch]
        )
        loss_physics = 0.5 * torch.sum(residual_physics**2)

        # Data loss: ||u_measured - u||²
        if has_measurements and config.alpha_data > 0:
            residual_data = measured_disp_torch - u[measured_dofs_torch]
            loss_data = torch.mean(residual_data**2)
            # Total weighted loss (with data term)
            loss_total = (
                config.alpha_physics * loss_physics + config.alpha_data * loss_data
            )
        else:
            loss_data = torch.tensor(0.0, dtype=torch.float32)
            # Total weighted loss (physics only, avoid NaN from 0*NaN)
            loss_total = config.alpha_physics * loss_physics

        # ========================================
        # Backpropagation
        # ========================================

        loss_total.backward()

        # Update variables
        optimizer_u.step()
        if optimizer_theta:
            optimizer_theta.step()

        # Enforce boundary conditions (u=0 on fixed DOFs)
        with torch.no_grad():
            u[fixed_dofs_torch] = 0.0

        # ========================================
        # Monitoring and convergence
        # ========================================

        u_norm = torch.norm(u[free_dofs_torch]).item()
        loss_val = loss_total.item()
        residual_norm = torch.norm(residual_physics).item()

        hist_entry = {
            "iteration": float(iteration + 1),
            "loss_total": loss_val,
            "loss_physics": loss_physics.item(),
            "loss_data": loss_data.item() if has_measurements else 0.0,
            "u_norm": u_norm,
            "residual_norm": residual_norm,
        }

        # Track NN parameter norms if applicable
        if theta_list:
            theta_norm = sum(torch.norm(p).item() for p in theta_list)
            hist_entry["theta_norm"] = theta_norm

        history.append(hist_entry)

        # Print progress
        if (iteration + 1) % config.print_every == 0 or iteration == 0:
            msg = (
                f"{iteration+1:6d} | {loss_val:12.3e} | {loss_physics.item():12.3e} | "
                f"{residual_norm:12.3e} | "
                f"{loss_data.item() if has_measurements else 0.0:12.3e} | {u_norm:10.3e}"
            )
            if has_nn:
                msg += f" | {theta_norm:10.3e}"
            print(msg)

        # Save best solution
        if not np.isnan(loss_val) and loss_val < best_loss:
            best_loss = loss_val

        # Convergence check: Use residual norm instead of loss (more robust)
        # For pure physics problems (no data), loss_total may have numerical issues
        if iteration > 10:
            # Primary criterion: residual norm (physical equilibrium)
            if residual_norm < config.tolerance:
                converged = True
                print(
                    f"\n[CONVERGED] Reached equilibrium in {iteration+1} iterations (residual={residual_norm:.2e} < tol={config.tolerance:.1e})"
                )
                break
            # Secondary criterion: loss (if available and not NaN)
            if not np.isnan(loss_val) and loss_val < config.tolerance:
                converged = True
                print(
                    f"\n[CONVERGED] Loss minimized in {iteration+1} iterations (loss={loss_val:.2e} < tol={config.tolerance:.1e})"
                )
                break

    if not converged:
        print(f"\n[WARNING] Did not converge in {config.max_iterations} iterations.")
        print(f"          Final loss: {loss_val:.3e}")

    # ========================================
    # Prepare result
    # ========================================

    # Final displacements
    with torch.no_grad():
        u_final = u.numpy()
        if model.dimension == 1:
            displacements_out = u_final.reshape(-1, 1)
        else:
            displacements_out = u_final.reshape(model.nnode, model.dimension)

    # Compute reactions (R = K*u - F_ext)
    with torch.no_grad():
        k_final, f_int_final = assemble_system_torch(
            model, u, load_factor=target_load_factor
        )
        reactions_torch = f_int_final - target_load_factor * f_ext_torch
        reactions_torch[free_dofs_torch] = 0.0  # Zero out free DOFs
        reactions_np = reactions_torch.numpy()

        if model.dimension == 1:
            reactions_out = reactions_np.reshape(-1, 1)
        else:
            reactions_out = reactions_np.reshape(model.nnode, model.dimension)

    # Extract NN parameters if present
    nn_params = None
    if theta_list:
        nn_params = {}
        for i, param in enumerate(theta_list):
            nn_params[f"param_{i}"] = param.detach().numpy()

    return SolverResult(
        displacements=displacements_out,
        reactions=reactions_out,
        converged=converged,
        history=history,
        nn_parameters=nn_params,
    )


# ============================================================================
# Newton-Raphson Solver (classical FEM, no NN)
# ============================================================================


def solve_nr(
    model: FEMModel,
    config: Optional[SolverConfig] = None,
    target_load_factor: float = 1.0,
    u_initial: Optional[torch.Tensor] = None,
) -> SolverResult:
    """Newton-Raphson solver for classical FEM (no NN parameters).

    Solves the nonlinear equilibrium equation:
        R(u) = F_int(u) - F_ext = 0

    Using Newton-Raphson iterations:
        u_{k+1} = u_k - K_tangent^{-1} * R(u_k)

    This is the classical FEM solver, equivalent to solve_incremental_newton
    from core.py. Only works with scalar material properties.

    Args:
        model: FEM model with scalar material properties
        config: Solver configuration
        target_load_factor: Target load factor for this solve (incremental loading)

    Returns:
        SolverResult with displacements, reactions, and convergence info
    """
    config = config or SolverConfig()

    # Check that model doesn't have NN parameters
    has_nn = model.material.has_trainable_params()
    if has_nn:
        raise ValueError(
            "Newton-Raphson solver with NN materials not fully supported yet. "
            "Use solve_gd() for problems with NN parameters."
        )

    u = np.zeros(model.ndof, dtype=float)
    free_dofs, fixed_dofs = free_and_fixed_dofs(model.ndof, model.fixed_dofs)
    history = []

    # Single increment with target_load_factor (loop now handled in solve())
    load_factor = target_load_factor
    f_ext = load_factor * model.loads

    has_converged = False
    residual_norm = np.inf
    max_e_gl = 0.0

    # Newton-Raphson iterations for this load factor
    for ite in range(config.max_iterations):
        k_tan, f_int, max_e_gl = assemble_system(model, u)
        rhs = f_ext - f_int

        k_ff = k_tan[np.ix_(free_dofs, free_dofs)]
        rhs_f = rhs[free_dofs]

        try:
            du_f = np.linalg.solve(k_ff, rhs_f)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Tangent stiffness became singular during solve"
            ) from exc

        du = np.zeros_like(u)
        du[free_dofs] = du_f
        du[fixed_dofs] = 0.0

        u += du
        residual_norm = np.linalg.norm(du) / max(
            np.linalg.norm(u), config.min_denominator
        )

        if residual_norm <= config.tolerance:
            has_converged = True
            break

    # Record history for this increment
    history.append(
        {
            "load_factor": float(load_factor),
            "iterations": float(ite + 1),
            "residual": float(residual_norm),
            "max_strain": float(max_e_gl),
            "converged": float(1.0 if has_converged else 0.0),
        }
    )

    # Compute final reactions
    k_final, f_int_final, _ = assemble_system(model, u)
    reactions = k_final @ u - load_factor * model.loads
    reactions[free_dofs] = 0.0

    # Format output
    if model.dimension == 1:
        displacements_out = u.reshape(-1, 1)
        reactions_out = reactions.reshape(-1, 1)
    else:
        displacements_out = u.reshape(model.nnode, model.dimension)
        reactions_out = reactions.reshape(model.nnode, model.dimension)

    return SolverResult(
        displacements=displacements_out,
        reactions=reactions_out,
        converged=has_converged,
        history=history,
    )


# ============================================================================
# Hybrid NR-GD Solver (TODO)
# ============================================================================


def solve_hybrid(
    model: FEMModel,
    config: Optional[SolverConfig] = None,
    measured_disp: Optional[np.ndarray] = None,
    measured_dofs: Optional[List[int]] = None,
    target_load_factor: float = 1.0,
    u_initial: Optional[torch.Tensor] = None,
) -> SolverResult:
    """Hybrid Newton-Raphson + Gradient Descent solver.

    Strategy:
    1. Phase 1: GD preconditioning (limited iterations, warm up NN parameters)
    2. Phase 2: NR finalization (fast convergence to tight tolerance)

    This combines the robustness of GD with the speed of NR.

    Args:
        model: FEM model with NN material properties
        config: Solver configuration
        measured_disp: Optional measured displacements
        measured_dofs: Optional DOF indices for measurements
        target_load_factor: Target load factor for this solve (incremental loading)
        u_initial: Optional warm start displacement vector

    Returns:
        SolverResult with displacements, reactions, and convergence info
    """
    config = config or SolverConfig()

    print(f"=== HYBRID SOLVER ===")
    print(f"Target load factor: {target_load_factor}")

    # Phase 1: GD Preconditioning (only if enabled)
    gd_result = None
    if config.preconditioning:
        print("Phase 1: GD Preconditioning...")
        gd_config = copy.deepcopy(config)
        gd_config.max_iterations = min(
            300, config.max_iterations // 3
        )  # Limited GD iterations
        gd_config.tolerance = max(
            1e-4, config.tolerance * 10
        )  # Relaxed tolerance for GD phase

        try:
            gd_result = solve_gd(
                model,
                gd_config,
                measured_disp,
                measured_dofs,
                target_load_factor,
                u_initial,
                skip_preconditioning=True,
            )
            print(f"  GD Phase: {gd_result.history[-1]['iteration']} iterations")

            # If GD already converged to tight tolerance, return it
            if (
                gd_result.converged
                and gd_result.history[-1].get("residual_norm", 1.0) < config.tolerance
            ):
                print(f"  GD achieved tight convergence, skipping NR phase")
                return gd_result

        except Exception as e:
            print(f"  GD Phase failed: {e}, proceeding with cold NR")
            gd_result = None
    else:
        print("Phase 1: GD Preconditioning SKIPPED (preconditioning=False)")

    # Phase 2: Newton-Raphson Finalization
    print("Phase 2: Newton-Raphson Finalization...")

    # Check if we can use NR (requires scalar material properties)
    has_nn = model.material.has_trainable_params()
    if has_nn:
        print(
            f"  NN parameters detected. Using GD for final convergence with tight tolerance."
        )
        # Use GD but with tight tolerance for final convergence
        final_config = copy.deepcopy(config)
        final_config.max_iterations = config.max_iterations - (
            gd_config.max_iterations if gd_result else 0
        )
        final_config.tolerance = config.tolerance  # Original tight tolerance

        # Use GD result as warm start if available
        u_warm = (
            torch.tensor(gd_result.displacements.flatten(), dtype=torch.float32)
            if gd_result
            else u_initial
        )
        final_result = solve_gd(
            model,
            final_config,
            measured_disp,
            measured_dofs,
            target_load_factor,
            u_warm,
            skip_preconditioning=True,
        )

        # Combine iteration counts
        if gd_result:
            gd_iterations = (
                gd_result.history[-1].get("iteration", 0) if gd_result.history else 0
            )
            final_iterations = (
                final_result.history[-1].get("iteration", 0)
                if final_result.history
                else 0
            )
            total_iterations = gd_iterations + final_iterations

            # Merge histories with proper iteration numbering
            unified_history = []
            if gd_result.history:
                unified_history.extend(gd_result.history)

            if final_result.history:
                for entry in final_result.history:
                    new_entry = entry.copy()
                    new_entry["iteration"] = entry.get("iteration", 0) + gd_iterations
                    unified_history.append(new_entry)

            final_result.history = unified_history

        total_iterations = (
            final_result.history[-1].get("iteration", 0) if final_result.history else 0
        )
        print(f"  Hybrid Total: {total_iterations} iterations")
        return final_result

    else:
        # Pure scalar materials: can use true NR
        print(f"  Scalar materials detected. Using Newton-Raphson.")
        u_warm = (
            torch.tensor(gd_result.displacements.flatten(), dtype=torch.float32)
            if gd_result
            else u_initial
        )
        nr_result = solve_nr(model, config, target_load_factor, u_warm)

        # NR uses different history format: check what's available
        nr_iterations = (
            nr_result.history[-1].get("iterations", 1) if nr_result.history else 1
        )
        print(f"  NR Phase: {nr_iterations} iterations")

        # Combine results - handle different history formats
        if gd_result:
            gd_iterations = (
                gd_result.history[-1].get("iteration", 0) if gd_result.history else 0
            )
            total_iterations = gd_iterations + nr_iterations

            # Create unified history format
            unified_history = []
            if gd_result.history:
                unified_history.extend(gd_result.history)

            # Add NR phase to history (convert NR format to GD format)
            if nr_result.history:
                nr_entry = nr_result.history[-1].copy()
                nr_entry["iteration"] = total_iterations  # Use unified iteration count
                unified_history.append(nr_entry)

            nr_result.history = unified_history

        print(
            f"  Hybrid Total: {nr_result.history[-1].get('iteration', nr_iterations)} iterations"
        )
        return nr_result


# ============================================================================
# Auxiliary Functions
# ============================================================================


def _compute_loss_components(
    u: torch.Tensor,
    model: FEMModel,
    load_factor: float,
    measured_disp_torch: Optional[torch.Tensor] = None,
    measured_dofs: Optional[List[int]] = None,
    alpha_physics: float = 1.0,
    alpha_data: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute physics and data loss components for Full Newton-Raphson.

    Args:
        u: Current displacements (torch tensor)
        model: FEM model
        load_factor: Current load scaling factor
        measured_disp_torch: Measured displacement data (if any)
        measured_dofs: DOF indices for measurements
        alpha_physics: Weight for physics loss
        alpha_data: Weight for data fitting loss

    Returns:
        Tuple of (loss_physics, loss_data)
    """
    # Compute physics residual: F_int(u) - load_factor * F_ext
    f_int_torch, _ = assemble_system_torch(model, u)
    f_ext_torch = torch.tensor(model.loads, dtype=torch.float32)
    residual_physics = f_int_torch - load_factor * f_ext_torch

    # Physics loss: ||R(u)||^2
    loss_physics = 0.5 * torch.sum(residual_physics**2)

    # Data fitting loss (if measurements provided)
    has_measurements = (
        measured_disp_torch is not None
        and measured_dofs is not None
        and len(measured_dofs) > 0
    )

    if has_measurements and alpha_data > 0:
        measured_dofs_torch = torch.tensor(measured_dofs, dtype=torch.long)
        residual_data = u[measured_dofs_torch] - measured_disp_torch
        loss_data = torch.mean(residual_data**2)
    else:
        loss_data = torch.tensor(0.0, dtype=torch.float32)

    return loss_physics, loss_data


# ============================================================================
# Full Newton-Raphson Solver with Hessian (TODO)
# ============================================================================


def solve_full_nr(
    model: FEMModel,
    config: Optional[SolverConfig] = None,
    measured_disp: Optional[np.ndarray] = None,
    measured_dofs: Optional[List[int]] = None,
    target_load_factor: float = 1.0,
) -> SolverResult:
    """Full Newton-Raphson solver with complete Hessian.

    Solves the coupled system:
        [H_uu  H_uθ] [Δu]   = - [∂L/∂u]
        [H_θu  H_θθ] [Δθ]       [∂L/∂θ]

    For problems without NN parameters (θ = ∅), this reduces to classical Newton-Raphson:
        H_uu * Δu = -∂L/∂u  →  K_tan * Δu = -R(u)

    Args:
        model: FEM model (with or without NN material properties)
        config: Solver configuration
        measured_disp: Optional measured displacements
        measured_dofs: Optional DOF indices for measurements
        target_load_factor: Target load factor for this solve (incremental loading)

    Returns:
        SolverResult with displacements, parameters, and convergence info

    Where H is the Hessian matrix (second derivatives).
    Computationally expensive but has quadratic convergence.
    """
    config = config or SolverConfig()

    # Check if model has NN parameters
    has_nn = model.material.has_trainable_params()

    if not has_nn:
        # For problems without NN parameters, Full NR = Classical NR
        # Delegate to solve_nr for efficiency
        return solve_nr(model, config, target_load_factor)

    # Convert to PyTorch for NN parameter optimization
    if not isinstance(model.nodes, torch.Tensor):
        model.nodes = torch.tensor(
            model.nodes, dtype=torch.float32, requires_grad=False
        )
    if not isinstance(model.loads, torch.Tensor):
        model.loads = torch.tensor(
            model.loads, dtype=torch.float32, requires_grad=False
        )

    # Initialize variables
    u = torch.zeros(model.ndof, dtype=torch.float32, requires_grad=True)
    free_dofs, fixed_dofs = free_and_fixed_dofs(model.ndof, model.fixed_dofs)

    # Get NN parameters
    nn_params = model.material.get_all_torch_params()
    if nn_params:
        # Ensure parameters require gradients
        for param in nn_params:
            param.requires_grad_(True)

    # Set up optimizer for Full Newton-Raphson with Hessian
    # For Full NR, we manually compute Hessian instead of using optimizer step

    history = []
    has_converged = False
    loss_total = torch.tensor(float("inf"))

    # Extract measured data setup
    measured_disp_torch = None
    if measured_disp is not None:
        measured_disp_torch = torch.tensor(measured_disp, dtype=torch.float32)

    print(f"=== FULL NEWTON-RAPHSON WITH HESSIAN ===")
    print(f"Target load factor: {target_load_factor}")
    print(f"NN parameters: {len(nn_params)} tensors")
    print(f"Measured DOFs: {measured_dofs if measured_dofs else 'None'}")

    # Full Newton-Raphson iterations
    for iteration in range(config.max_iterations):

        # Compute loss and its gradients
        load_factor = target_load_factor
        loss_physics, loss_data = _compute_loss_components(
            u,
            model,
            load_factor,
            measured_disp_torch,
            measured_dofs,
            config.alpha_physics,
            config.alpha_data,
        )

        loss_total = loss_physics + loss_data

        # Compute first-order gradients ∂L/∂u and ∂L/∂θ
        grad_u = torch.autograd.grad(
            loss_total, u, create_graph=True, retain_graph=True, allow_unused=True
        )[0]

        # Handle case where u is not in computation graph
        if grad_u is None:
            grad_u = torch.zeros_like(u)

        grad_theta = []
        if nn_params:
            for param in nn_params:
                grad = torch.autograd.grad(
                    loss_total,
                    param,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if grad is not None:
                    grad_theta.append(grad.flatten())
            grad_theta = (
                torch.cat(grad_theta)
                if grad_theta
                else torch.tensor([], dtype=torch.float32)
            )

        # Check convergence on gradients (Full NR criterion)
        grad_norm_u = torch.norm(grad_u[free_dofs]).item()
        grad_norm_theta = torch.norm(grad_theta).item() if len(grad_theta) > 0 else 0.0
        total_grad_norm = np.sqrt(grad_norm_u**2 + grad_norm_theta**2)

        # Print progress
        if iteration % config.print_every == 0 or iteration < 5:
            nn_param_norms = (
                [torch.norm(p).item() for p in nn_params] if nn_params else []
            )
            nn_summary = f"{np.mean(nn_param_norms):.3e}" if nn_param_norms else "N/A"

            print(
                f"{iteration:4d} | {loss_total.item():12.3e} | {loss_physics.item():12.3e} | "
                f"{grad_norm_u:12.3e} | {loss_data.item():12.3e} | "
                f"{torch.norm(u).item():10.3e} | {nn_summary:>10}"
            )

        # Convergence check
        if total_grad_norm < config.tolerance:
            has_converged = True
            print(
                f"[CONVERGED] Gradients minimized in {iteration} iterations "
                f"(||∇L|| = {total_grad_norm:.2e} < tol = {config.tolerance})"
            )
            break

        # Compute Hessian matrix for Full Newton-Raphson step
        # This is computationally expensive but provides quadratic convergence
        try:
            # Compute Hessian blocks: H_uu, H_uθ, H_θu, H_θθ
            n_u = len(free_dofs)
            n_theta = len(grad_theta)

            if n_theta == 0:
                # No NN parameters: H = H_uu only
                H_uu = torch.zeros(n_u, n_u)
                for i in range(n_u):
                    grad2 = torch.autograd.grad(
                        grad_u[free_dofs[i]], u, retain_graph=True
                    )[0]
                    H_uu[i] = grad2[free_dofs]

                # Solve H_uu * Δu = -grad_u
                try:
                    delta_u = torch.linalg.solve(H_uu, -grad_u[free_dofs])
                    u[free_dofs] += delta_u
                except torch.linalg.LinAlgError:
                    # Fallback to pseudoinverse for singular matrices
                    delta_u = torch.linalg.pinv(H_uu) @ (-grad_u[free_dofs])
                    u[free_dofs] += delta_u

            else:
                # Full coupled system with NN parameters
                # This is very expensive - typically fallback to hybrid solver
                print(
                    f"  [{iteration}] Computing full Hessian {n_u}×{n_u} + {n_theta}×{n_theta}..."
                )

                # For now, use a simplified approach: alternate updates
                # (True Full NR with complete Hessian is extremely expensive)

                # Update displacements with H_uu^{-1} * grad_u
                H_uu = torch.zeros(n_u, n_u)
                for i in range(min(n_u, 20)):  # Limit for computational efficiency
                    if i < n_u:
                        grad2 = torch.autograd.grad(
                            grad_u[free_dofs[i]], u, retain_graph=True
                        )[0]
                        H_uu[i] = grad2[free_dofs]

                # Regularization for stability
                H_uu += 1e-8 * torch.eye(n_u)

                try:
                    delta_u = torch.linalg.solve(H_uu, -grad_u[free_dofs])
                    u[free_dofs] += 0.5 * delta_u  # Damped update
                except torch.linalg.LinAlgError:
                    delta_u = torch.linalg.pinv(H_uu) @ (-grad_u[free_dofs])
                    u[free_dofs] += 0.1 * delta_u  # Very conservative

                # Update NN parameters (simplified - using gradient step)
                with torch.no_grad():
                    idx = 0
                    for param in nn_params:
                        param_size = param.numel()
                        param_grad = grad_theta[idx : idx + param_size].reshape(
                            param.shape
                        )
                        param -= 0.01 * param_grad  # Small step for stability
                        idx += param_size

        except Exception as e:
            print(f"  [WARNING] Hessian computation failed: {e}")
            print(f"  [FALLBACK] Using gradient descent step...")

            # Fallback to gradient descent
            with torch.no_grad():
                u[free_dofs] -= config.learning_rate_u * grad_u[free_dofs]

                if nn_params:
                    idx = 0
                    for param in nn_params:
                        param_size = param.numel()
                        param_grad = grad_theta[idx : idx + param_size].reshape(
                            param.shape
                        )
                        param -= config.learning_rate_theta * param_grad
                        idx += param_size

        # Store history
        history.append(
            {
                "iteration": iteration,
                "loss_total": loss_total.item(),
                "loss_physics": loss_physics.item(),
                "loss_data": loss_data.item(),
                "grad_norm_u": grad_norm_u,
                "grad_norm_theta": grad_norm_theta,
            }
        )

    # Final status
    if not has_converged:
        print(f"[MAX ITER] Reached maximum iterations ({config.max_iterations})")
        print(f"  Final gradient norm: {total_grad_norm:.2e}")

    # Return results
    u_final = u.detach().numpy()

    # Format displacements
    if model.dimension == 1:
        displacements_out = u_final.reshape(-1, 1)
    else:
        displacements_out = u_final.reshape(model.nnode, model.dimension)

    # Compute reactions (R = F_int - F_ext)
    with torch.no_grad():
        k_final, f_int_final = assemble_system_torch(
            model, u, load_factor=target_load_factor
        )
        reactions_torch = f_int_final - target_load_factor * f_ext_torch
        reactions_torch[free_dofs_torch] = 0.0  # Zero out free DOFs
        reactions_np = reactions_torch.numpy()

        if model.dimension == 1:
            reactions_out = reactions_np.reshape(-1, 1)
        else:
            reactions_out = reactions_np.reshape(model.nnode, model.dimension)

    # Extract NN parameters if present
    nn_parameters = None
    if nn_params:
        nn_parameters = {}
        for i, param in enumerate(nn_params):
            nn_parameters[f"param_{i}"] = param.detach().numpy()

    return SolverResult(
        displacements=displacements_out,
        reactions=reactions_out,
        converged=has_converged,
        history=history,
        nn_parameters=nn_parameters,
    )


# ============================================================================
# Auto-select solver based on problem type
# ============================================================================


def solve(
    model: FEMModel,
    config: Optional[SolverConfig] = None,
    measured_disp: Optional[np.ndarray] = None,
    measured_dofs: Optional[List[int]] = None,
) -> SolverResult:
    """Universal solver with incremental loading for all methods.

    Incremental loading strategy:
    - Divides load from initial to final in n_increments steps
    - Each increment converges before moving to next
    - Works for NR, GD, hybrid, with/without NN

    Args:
        model: FEM model
        config: Solver configuration (method can be "auto" or specific)
        measured_disp: Optional measured displacements
        measured_dofs: Optional DOF indices for measurements

    Returns:
        SolverResult from final increment
    """
    config = config or SolverConfig()

    # Determine solver method
    if config.method != "auto":
        method = config.method.lower()
    else:
        # Auto-selection logic
        has_nn = model.material.has_trainable_params()
        has_measurements = measured_disp is not None and measured_dofs is not None

        if not has_nn and not has_measurements:
            print("[AUTO] Selecting: Newton-Raphson (classical FEM)")
            method = "nr"
        elif has_nn:
            print("[AUTO] Selecting: Gradient Descent (PINN)")
            method = "gd"
        else:
            print("[AUTO] Selecting: Gradient Descent (inverse problem)")
            method = "gd"

    # Universal incremental loading loop
    print(f"\n{'Inc':>4} | {'Load Factor':>12} | {'Status':>10}")
    print("-" * 40)

    result = None
    u_current = None  # Carry displacements between increments

    for iinc in range(1, config.n_increments + 1):
        # Compute load factor for this increment
        load_factor = config.load_factor_initial + (iinc / config.n_increments) * (
            config.load_factor_final - config.load_factor_initial
        )

        # Warm start message
        if u_current is not None:
            print(f"{iinc:>4} | {load_factor:>12.4f} | {'WARM_START':>10}")
        else:
            print(f"{iinc:>4} | {load_factor:>12.4f} | {'COLD_START':>10}")

        # Convert u_current to torch tensor if available
        u_initial_torch = None
        if u_current is not None:
            # Ensure proper tensor conversion from numpy array
            u_initial_torch = torch.tensor(
                u_current, dtype=torch.float32, requires_grad=False
            )
            # Validate shape consistency
            print(
                f"    Warm start values: shape={u_initial_torch.shape}, range=[{u_initial_torch.min():.4f}, {u_initial_torch.max():.4f}]"
            )

        # Call appropriate solver for this increment with warm start
        if method == "gd":
            result = solve_gd(
                model,
                config,
                measured_disp,
                measured_dofs,
                target_load_factor=load_factor,
                u_initial=u_initial_torch,
            )
        elif method == "nr":
            result = solve_nr(
                model, config, target_load_factor=load_factor, u_initial=u_initial_torch
            )
        elif method == "hybrid":
            result = solve_hybrid(
                model,
                config,
                measured_disp,
                measured_dofs,
                target_load_factor=load_factor,
                u_initial=u_initial_torch,
            )
        elif method == "full-nr":
            result = solve_full_nr(
                model,
                config,
                measured_disp,
                measured_dofs,
                target_load_factor=load_factor,
            )
        else:
            raise ValueError(f"Unknown solver method: {method}")

        # Store displacements for next increment (keep as DOF vector, not node matrix)
        u_current = (
            result.displacements.flatten()
        )  # Convert from [nnode, ndim] to [ndof]

        # Print increment progress
        status = "CONVERGED" if result.converged else "FAILED"
        print(f"{iinc:4d} | {load_factor:12.6f} | {status:>10}")

        if not result.converged:
            print(
                f"[WARNING] Increment {iinc} did not converge, stopping incremental loading."
            )
            break

    return result
