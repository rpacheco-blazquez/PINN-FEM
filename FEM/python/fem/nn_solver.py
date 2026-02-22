"""Newton-Raphson multiparamétrico para problemas inversos PINN.

Resuelve simultáneamente desplazamientos [u] y parámetros NN [θ]:
    R_physics = K(θ)*u - F = 0      (equilibrio)
    R_data = u_measured - u = 0     (ajuste a mediciones)

Jacobiano completo:
    J = [ ∂R_physics/∂u    ∂R_physics/∂θ ]
        [ ∂R_data/∂u       ∂R_data/∂θ    ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from .model import FEMModel
from .boundary import free_and_fixed_dofs
from .nn_assembly import assemble_system_torch


@dataclass
class PINNSolverConfig:
    """Configuración del solver PINN."""
    max_iterations: int = 50
    tolerance: float = 1e-6
    alpha_physics: float = 1.0      # Peso del residual físico
    alpha_data: float = 1.0         # Peso del residual de datos
    min_denominator: float = 1e-12
    
    # Para controlar paso de actualización
    max_step_u: float = 1e-3        # Máximo cambio en desplazamientos [m]
    max_step_theta: float = 0.1     # Máximo cambio relativo en parámetros NN
    line_search: bool = True        # Usar backtracking line search


@dataclass
class PINNSolverResult:
    """Resultado del solver PINN."""
    displacements: np.ndarray       # (nnode, dim) desplazamientos finales
    nn_parameters: Dict[str, np.ndarray]  # Parámetros NN finales
    converged: bool
    history: List[Dict[str, float]]


def compute_jacobian_blocks(
    model: FEMModel,
    u: torch.Tensor,
    f_ext_torch: torch.Tensor,
    theta_list: List[torch.nn.Parameter],
    free_dofs: np.ndarray,
    measured_dofs: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Calcula los bloques del Jacobiano para Newton-Raphson PINN.
    
    Returns:
        j_uu: (n_free, n_free) - ∂R_physics/∂u = K(θ)
        j_utheta: (n_free, n_theta) - ∂R_physics/∂θ
        r_physics: (n_free,) - residual de equilibrio
        j_data_u: (n_measured, n_free) - ∂R_data/∂u (if measurements)
        r_data: (n_measured,) - residual de mediciones (if measurements)
    """
    n_free = len(free_dofs)
    n_params = sum(p.numel() for p in theta_list)
    
    # Habilitar gradientes
    u_active = u.detach().requires_grad_(True)
    
    # Ensamblar sistema
    k_global, f_int = assemble_system_torch(model, u_active)
    
    # Residual de física: R = f_int - f_ext
    r_physics_full = f_int - f_ext_torch
    r_physics = r_physics_full[free_dofs]
    
    # ========================================
    # Bloque J_uu = ∂R_physics/∂u = K(θ)
    # ========================================
    j_uu = k_global[free_dofs, :][:, free_dofs].detach()
    
    # ========================================
    # Bloque J_utheta = ∂R_physics/∂θ
    # ========================================
    # Calcular ∂(f_int[free_dofs])/∂θ usando autograd
    j_utheta_list = []
    
    for param in theta_list:
        # Calcular jacobiano para este parámetro: ∂r_physics/∂param
        param_grads = []
        
        for i in range(n_free):
            # Calcular ∂r_physics[i]/∂param
            if param.grad is not None:
                param.grad.zero_()
            
            # Backward para obtener gradiente de r_physics[i] respecto a param
            r_physics[i].backward(retain_graph=True)
            
            if param.grad is not None:
                param_grads.append(param.grad.detach().clone().flatten())
            else:
                param_grads.append(torch.zeros(param.numel(), dtype=torch.float32))
        
        # Apilar: (n_free, n_params_i)
        j_param = torch.stack(param_grads, dim=0)
        j_utheta_list.append(j_param)
    
    # Concatenar todos los parámetros: (n_free, n_params_total)
    if j_utheta_list:
        j_utheta = torch.cat(j_utheta_list, dim=1)
    else:
        j_utheta = torch.zeros((n_free, n_params), dtype=torch.float32)
    
    # ========================================
    # Bloques de mediciones (si existen)
    # ========================================
    if measured_dofs is not None:
        n_measured = len(measured_dofs)
        
        # J_data_u = ∂R_data/∂u = -I
        j_data_u = torch.zeros((n_measured, len(free_dofs)), dtype=torch.float32)
        for i, measured_dof in enumerate(measured_dofs):
            # Encontrar índice de measured_dof en free_dofs
            idx_in_free = np.where(free_dofs == measured_dof)[0]
            if len(idx_in_free) > 0:
                j_data_u[i, idx_in_free[0]] = -1.0
        
        # r_data se calcula fuera
        return j_uu, j_utheta, r_physics.detach(), j_data_u, None
    else:
        return j_uu, j_utheta, r_physics.detach(), None, None


def solve_pinn_newton_raphson(
    model: FEMModel,
    f_ext: np.ndarray,
    measured_disp: Optional[np.ndarray] = None,
    measured_dofs: Optional[List[int]] = None,
    config: Optional[PINNSolverConfig] = None,
) -> PINNSolverResult:
    """Solver Newton-Raphson para problema inverso PINN.
    
    Args:
        model: FEM model con material properties (deben contener NNProperty)
        f_ext: (ndof,) fuerzas externas aplicadas
        measured_disp: (n_measurements,) desplazamientos medidos
        measured_dofs: (n_measurements,) índices de DOFs donde hay mediciones
        config: configuración del solver
        
    Returns:
        PINNSolverResult con desplazamientos y parámetros NN optimizados
    """
    config = config or PINNSolverConfig()
    
    # Verificar que el modelo tiene parámetros entrenables
    if not model.material.has_trainable_params():
        raise ValueError("Model must have trainable NN parameters (use NNProperty)")
    
    # Obtener parámetros NN
    theta_list = model.material.get_all_torch_params()
    if not theta_list:
        raise ValueError("No trainable parameters found in model.material")
    
    # Inicializar desplazamientos
    u = torch.zeros(model.ndof, dtype=torch.float32, requires_grad=True)
    f_ext_torch = torch.tensor(f_ext, dtype=torch.float32)
    
    # DOFs libres y fijos
    free_dofs, fixed_dofs = free_and_fixed_dofs(model.ndof, model.fixed_dofs)
    n_free = len(free_dofs)
    
    # Mediciones (si existen)
    has_measurements = measured_disp is not None and measured_dofs is not None
    if has_measurements:
        measured_disp_torch = torch.tensor(measured_disp, dtype=torch.float32)
        measured_dofs = np.array(measured_dofs, dtype=int)
        n_measured = len(measured_dofs)
    else:
        n_measured = 0
    
    history = []
    converged = False
    
    n_params_total = sum(p.numel() for p in theta_list)
    
    print(f"\n{'='*70}")
    print(f"Newton-Raphson PINN: {n_free} DOFs libres, {n_params_total} parámetros NN")
    if has_measurements:
        print(f"Mediciones: {n_measured} DOFs")
    print(f"{'='*70}")
    print(f"{'Iter':>5} | {'||R_phys||':>12} | {'||R_data||':>12} | {'||R_tot||':>12} | {'Step':>6}")
    print("-" * 70)
    
    for iteration in range(config.max_iterations):
        # ========================================
        # 1. CALCULAR JACOBIANO Y RESIDUALES
        # ========================================
        
        # Calcular bloques del Jacobiano
        j_uu, j_utheta, r_physics, j_data_u, _ = compute_jacobian_blocks(
            model=model,
            u=u,
            f_ext_torch=f_ext_torch,
            theta_list=theta_list,
            free_dofs=free_dofs,
            measured_dofs=measured_dofs if has_measurements else None,
        )
        
        # Residual de mediciones
        if has_measurements:
            r_data = measured_disp_torch - u[measured_dofs]
        else:
            r_data = torch.zeros(0, dtype=torch.float32)
        
        # ========================================
        # 2. ENSAMBLAR JACOBIANO Y RESIDUAL COMPLETOS
        # ========================================
        
        if has_measurements:
            # J_data_theta = 0 (mediciones no dependen de θ)
            j_data_theta = torch.zeros((n_measured, n_params_total), dtype=torch.float32)
            
            # J = [ α*J_uu        α*J_utheta    ]
            #     [ β*J_data_u    β*J_data_theta ]
            j_top = torch.cat([
                config.alpha_physics * j_uu,
                config.alpha_physics * j_utheta
            ], dim=1)
            j_bottom = torch.cat([
                config.alpha_data * j_data_u,
                config.alpha_data * j_data_theta
            ], dim=1)
            jacobian = torch.cat([j_top, j_bottom], dim=0)
            
            # R = [ α*r_physics ]
            #     [ β*r_data    ]
            residual = torch.cat([
                config.alpha_physics * r_physics,
                config.alpha_data * r_data
            ])
        else:
            # Solo física
            jacobian = torch.cat([j_uu, j_utheta], dim=1)
            residual = config.alpha_physics * r_physics
        
        # Normas de residuales
        r_physics_norm = torch.norm(r_physics).item()
        r_data_norm = torch.norm(r_data).item() if has_measurements else 0.0
        r_total_norm = torch.norm(residual).item()
        
        # ========================================
        # 3. RESOLVER SISTEMA (Gauss-Newton para underdetermined systems)
        # ========================================
        # 
        # Para problemas inversos con más variables que ecuaciones,
        # usamos Gauss-Newton: minimizar ||R||²
        #
        # ∇J = J^T R
        # H ≈ J^T J (Jacobiano aproximado del Hessiano)
        # Δx = -(J^T J)^{-1} J^T R
        
        try:
            # Formar sistema normal: (J^T J) Δx = -J^T R
            jt = jacobian.T
            jt_j = jt @ jacobian
            jt_r = jt @ residual
            
            # Regularización para estabilidad (Levenberg-Marquardt damping)
            damping = 1e-6 * torch.trace(jt_j) / jt_j.shape[0]
            jt_j_reg = jt_j + damping * torch.eye(jt_j.shape[0], dtype=torch.float32)
            
            # Resolver: (J^T J) Δx = -J^T R
            delta_x = torch.linalg.solve(jt_j_reg, -jt_r)
            
        except RuntimeError as e:
            print(f"\n⚠️  Solver falló at iteration {iteration+1}")
            print(f"    Error: {e}")
            print(f"    J shape: {jacobian.shape}, rank: {torch.linalg.matrix_rank(jacobian).item()}")
            break
        
        # Separar Δu y Δθ
        delta_u_free = delta_x[:n_free]
        delta_theta_flat = delta_x[n_free:]
        
        # ========================================
        # 4. LINE SEARCH (backtracking con Armijo condition mejorada)
        # ========================================
        
        step_size = 1.0
        armijo_c = 1e-4  # Constante de Armijo
        
        if config.line_search:
            best_norm = r_total_norm
            gradient_norm = torch.norm(jt @ residual).item()
            
            for ls_iter in range(15):
                # Probar paso
                u_test = u.clone().detach()
                u_test[free_dofs] += step_size * delta_u_free
                u_test[fixed_dofs] = 0.0
                
                # Actualizar θ temporalmente
                theta_backup = [p.data.clone() for p in theta_list]
                offset = 0
                for param in theta_list:
                    n_p = param.numel()
                    delta_p = delta_theta_flat[offset:offset+n_p].reshape(param.shape)
                    param.data += step_size * delta_p
                    offset += n_p
                
                # Evaluar nuevo residual
                try:
                    k_test, f_int_test = assemble_system_torch(model, u_test)
                    r_phys_test = f_int_test[free_dofs] - f_ext_torch[free_dofs]
                    
                    if has_measurements:
                        r_data_test = measured_disp_torch - u_test[measured_dofs]
                        r_test = torch.cat([
                            config.alpha_physics * r_phys_test,
                            config.alpha_data * r_data_test
                        ])
                    else:
                        r_test = config.alpha_physics * r_phys_test
                    
                    r_test_norm = torch.norm(r_test).item()
                    
                    # Armijo condition: f(x+αp) ≤ f(x) + c*α*<∇f, p>
                    # Simplificado: acepta si reduce significativamente el residual
                    if r_test_norm < best_norm * (1.0 - armijo_c * step_size):
                        break
                    
                except Exception as e:
                    # Si falla el ensamblaje, reducir paso
                    pass
                
                # Restaurar θ y reducir step
                for param, backup in zip(theta_list, theta_backup):
                    param.data.copy_(backup)
                step_size *= 0.7  # Más agresivo que 0.5
                
                if step_size < 1e-10:
                    print(f"    Line search mínimo alcanzado")
                    for param, backup in zip(theta_list, theta_backup):
                        param.data.copy_(backup)
                    step_size = 0.0
                    break
            
            # Si line search no mejoró NADA después de muchos intentos, usar paso pequeño
            if step_size < 1e-8 and step_size > 0:
                print(f"    Warning: Line search estancado, forzando paso pequeño")
                step_size = 1e-6
        
        # ========================================
        # 5. ACTUALIZAR VARIABLES
        # ========================================
        
        if step_size > 0:
            with torch.no_grad():
                u[free_dofs] += step_size * delta_u_free
                u[fixed_dofs] = 0.0
                
                offset = 0
                for param in theta_list:
                    n_p = param.numel()
                    delta_p = delta_theta_flat[offset:offset+n_p].reshape(param.shape)
                    param.data += step_size * delta_p
                    offset += n_p
        
        # ========================================
        # 6. CONVERGENCIA
        # ========================================
        
        relative_error = r_total_norm / max(torch.norm(u[free_dofs]).item(), config.min_denominator)
        
        history.append({
            "iteration": float(iteration + 1),
            "r_physics": r_physics_norm,
            "r_data": r_data_norm,
            "r_total": r_total_norm,
            "relative_error": relative_error,
            "step_size": float(step_size),
        })
        
        # Imprimir progreso
        print(f"{iteration+1:5d} | {r_physics_norm:12.3e} | {r_data_norm:12.3e} | "
              f"{r_total_norm:12.3e} | {step_size:6.3f}")
        
        if relative_error < config.tolerance and step_size > 0:
            converged = True
            print(f"\n✅ Convergió en {iteration+1} iteraciones!")
            break
        
        if step_size == 0.0:
            print(f"\n⚠️  Line search falló - deteniendo iteraciones")
            break
    
    if not converged:
        print(f"\n⚠️  No convergió en {config.max_iterations} iteraciones")
        print(f"    Residual final: {r_total_norm:.3e}")
    
    # ========================================
    # 7. PREPARAR RESULTADO
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
    
    return PINNSolverResult(
        displacements=displacements_out,
        nn_parameters=nn_params,
        converged=converged,
        history=history,
    )

