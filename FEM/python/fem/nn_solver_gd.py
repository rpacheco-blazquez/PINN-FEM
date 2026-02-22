"""Solver PINN simplificado usando gradient descent.

Más simple y robusto que Newton-Raphson completo.
Minimiza: J(u, θ) = ||K(θ)*u - F||² + λ||u_measured - u||²
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np

from .model import FEMModel
from .boundary import free_and_fixed_dofs
from .nn_assembly import assemble_system_torch


@dataclass
class PINNGradientDescentConfig:
    """Configuración del solver PINN con gradient descent."""
    max_iterations: int = 1000
    tolerance: float = 1e-6
    learning_rate_u: float = 1e-7      # Learning rate para desplazamientos
    learning_rate_theta: float = 1e-4  # Learning rate para parámetros NN
    alpha_physics: float = 1.0         # Peso del residual físico
    alpha_data: float = 100.0          # Peso del ajuste a datos
    print_every: int = 10              # Imprimir cada N iteraciones


@dataclass
class PINNGradientDescentResult:
    """Resultado del solver PINN."""
    displacements: np.ndarray
    nn_parameters: Dict[str, np.ndarray]
    converged: bool
    history: List[Dict[str, float]]


def solve_pinn_gradient_descent(
    model: FEMModel,
    f_ext: np.ndarray,
    measured_disp: Optional[np.ndarray] = None,
    measured_dofs: Optional[List[int]] = None,
    config: Optional[PINNGradientDescentConfig] = None,
) -> PINNGradientDescentResult:
    """Solver PINN usando gradient descent.
    
    Minimiza la loss function:
        J(u, θ) = α||K(θ)*u - F||² + β||u_measured - u||²
    
    Args:
        model: FEM model con material properties (deben contener NNProperty)
        f_ext: (ndof,) fuerzas externas aplicadas
        measured_disp: (n_measurements,) desplazamientos medidos
        measured_dofs: (n_measurements,) índices de DOFs donde hay mediciones
        config: configuración del solver
        
    Returns:
        PINNGradientDescentResult con desplazamientos y parámetros optimizados
    """
    config = config or PINNGradientDescentConfig()
    
    # Verificar que el modelo tiene parámetros entrenables
    if not model.material.has_trainable_params():
        raise ValueError("Model must have trainable NN parameters (use NNProperty)")
    
    # Obtener parámetros NN
    theta_list = model.material.get_all_torch_params()
    if not theta_list:
        raise ValueError("No trainable parameters found in model.material")
    
    # Inicializar desplazamientos (con gradientes)
    u = torch.zeros(model.ndof, dtype=torch.float32, requires_grad=True)
    f_ext_torch = torch.tensor(f_ext, dtype=torch.float32)
    
    # DOFs libres y fijos
    free_dofs, fixed_dofs = free_and_fixed_dofs(model.ndof, model.fixed_dofs)
    free_dofs_torch = torch.tensor(free_dofs, dtype=torch.long)
    
    # Mediciones (si existen)
    has_measurements = measured_disp is not None and measured_dofs is not None
    if has_measurements:
        measured_disp_torch = torch.tensor(measured_disp, dtype=torch.float32)
        measured_dofs_torch = torch.tensor(measured_dofs, dtype=torch.long)
    
    # Optimizadores separados para u y θ
    optimizer_u = torch.optim.Adam([u], lr=config.learning_rate_u)
    optimizer_theta = torch.optim.Adam(theta_list, lr=config.learning_rate_theta)
    
    history = []
    converged = False
    best_loss = float('inf')
    
    print(f"{'Iter':>6} | {'Loss Total':>12} | {'Loss Physics':>12} | {'Loss Data':>12} | {'||u||':>10}")
    print("-" * 70)
    
    for iteration in range(config.max_iterations):
        # Zero gradients
        optimizer_u.zero_grad()
        optimizer_theta.zero_grad()
        
        # ========================================
        # Ensamblar sistema y calcular residuales
        # ========================================
        
        k_global, f_int = assemble_system_torch(model, u)
        
        # Loss término de equilibrio: ||K*u - F||² (solo en DOFs libres)
        residual_physics = f_int[free_dofs_torch] - f_ext_torch[free_dofs_torch]
        loss_physics = torch.mean(residual_physics ** 2)
        
        # Loss término de datos: ||u_measured - u||²
        if has_measurements:
            residual_data = measured_disp_torch - u[measured_dofs_torch]
            loss_data = torch.mean(residual_data ** 2)
        else:
            loss_data = torch.tensor(0.0, dtype=torch.float32)
        
        # Loss total ponderada
        loss_total = (
            config.alpha_physics * loss_physics +
            config.alpha_data * loss_data
        )
        
        # ========================================
        # Backpropagation
        # ========================================
        
        loss_total.backward()
        
        # Actualizar variables
        optimizer_u.step()
        optimizer_theta.step()
        
        # Imponer condiciones de contorno (u=0 en DOFs fijos)
        with torch.no_grad():
            u[fixed_dofs] = 0.0
        
        # ========================================
        # Monitoreo y convergencia
        # ========================================
        
        u_norm = torch.norm(u[free_dofs_torch]).item()
        loss_val = loss_total.item()
        
        history.append({
            "iteration": float(iteration + 1),
            "loss_total": loss_val,
            "loss_physics": loss_physics.item(),
            "loss_data": loss_data.item() if has_measurements else 0.0,
            "u_norm": u_norm,
        })
        
        # Imprimir progreso
        if (iteration + 1) % config.print_every == 0 or iteration == 0:
            print(f"{iteration+1:6d} | {loss_val:12.3e} | {loss_physics.item():12.3e} | "
                  f"{loss_data.item() if has_measurements else 0.0:12.3e} | {u_norm:10.3e}")
        
        # Guardar mejor solución
        if loss_val < best_loss:
            best_loss = loss_val
        
        # Convergencia
        if iteration > 10 and loss_val < config.tolerance:
            converged = True
            print(f"\n✅ Convergió en {iteration+1} iteraciones!")
            break
    
    if not converged:
        print(f"\n⚠️  No convergió en {config.max_iterations} iteraciones.")
        print(f"   Loss final: {loss_val:.3e}")
    
    # ========================================
    # Preparar resultado
    # ========================================
    
    # Desplazamientos finales
    with torch.no_grad():
        if model.dimension == 1:
            displacements_out = u.numpy().reshape(-1, 1)
        else:
            displacements_out = u.numpy().reshape(model.nnode, model.dimension)
    
    # Parámetros NN finales
    nn_params = {}
    for i, param in enumerate(theta_list):
        nn_params[f"param_{i}"] = param.detach().numpy()
    
    return PINNGradientDescentResult(
        displacements=displacements_out,
        nn_parameters=nn_params,
        converged=converged,
        history=history,
    )
