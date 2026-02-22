"""Ejemplo de problema inverso PINN usando Newton-Raphson multiparamétrico.

Mismo problema que pinn_inverse_problem.py pero usando Newton-Raphson
en lugar de gradient descent.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from fem import (
    FEMModel, Material, SolverConfig, solve_incremental_newton,
    NNProperty, solve_pinn_newton_raphson, PINNSolverConfig
)


class YoungModulusNet(nn.Module):
    """Red neuronal para E(x)."""
    
    def __init__(self, hidden_size: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        
        # Inicialización: queremos salida cercana a 1.0 (pre-softplus)
        with torch.no_grad():
            self.net[-1].bias.fill_(1.0)
            self.net[-1].weight.fill_(0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1) -> out: (batch, 1)"""
        return self.net(x)


def young_modulus_real(x: float, E0: float = 2.1e11, L: float = 3.0) -> float:
    """E(x) real (lineal): E = E0 * (1 + 0.5*x/L)"""
    return E0 * (1.0 + 0.5 * x / L)


def main():
    print("=" * 60)
    print("PROBLEMA INVERSO PINN: Newton-Raphson Multiparamétrico")
    print("=" * 60)
    
    # Parámetros
    E0 = 2.1e11  # Pa
    A = 1e-4     # m²
    L = 3.0      # m longitud total
    F = 1000.0   # N fuerza aplicada
    
    # ========================================
    # PASO 1: Generar "mediciones" con E(x) real
    # ========================================
    print("\n1️⃣  Generando mediciones con E(x) real...")
    
    nodes = np.array([0.0, 1.0, 2.0, 3.0])  # 4 nodos
    elements = [(0, 1), (1, 2), (2, 3)]      # 3 elementos
    fixed_dofs = [0, 3]                       # Empotrado en ambos extremos
    loads = np.array([0.0, F, 0.0, 0.0])     # Fuerza en nodo 1
    
    # Modelo con E constante para generar mediciones sintéticas
    model_real = FEMModel(
        nodes=nodes,
        elements=elements,
        material=Material(young=E0, area=A, density=7800.0),
        fixed_dofs=fixed_dofs,
        loads=loads,
        dimension=1,
    )
    
    result_real = solve_incremental_newton(model_real, SolverConfig())
    u_measured = result_real.displacements.flatten()
    
    print(f"   Desplazamientos 'medidos':")
    for i, u in enumerate(u_measured):
        print(f"      Nodo {i}: u = {u:.6e} m")
    
    # ========================================
    # PASO 2: Definir modelo PINN con E(x) = NN(x)
    # ========================================
    print("\n2️⃣  Creando modelo PINN con E(x) = NN(x)...")
    
    young_net = YoungModulusNet(hidden_size=10)
    young_nn_property = NNProperty(
        net=young_net,
        input_dim=1,
        enforce_positive=True,
        scale=E0,
    )
    
    model_pinn = FEMModel(
        nodes=nodes,
        elements=elements,
        material=Material(
            young=young_nn_property,
            area=A,
            density=7800.0,
        ),
        fixed_dofs=fixed_dofs,
        loads=loads,
        dimension=1,
    )
    
    print(f"   Red neuronal creada:")
    print(f"      Parámetros entrenables: {sum(p.numel() for p in young_net.parameters())}")
    print(f"      Arquitectura: 1 → 10 → 1")
    
    # ========================================
    # PASO 3: Resolver con Newton-Raphson PINN
    # ========================================
    print("\n3️⃣  Resolviendo con Newton-Raphson multiparamétrico...")
    
    measured_dofs = [1, 2]
    measured_values = u_measured[measured_dofs]
    
    config_pinn = PINNSolverConfig(
        max_iterations=50,
        tolerance=1e-7,
        alpha_physics=0.1,    # Reducir peso de física
        alpha_data=10.0,      # Aumentar peso de datos
        line_search=True,
    )
    
    try:
        result_pinn = solve_pinn_newton_raphson(
            model=model_pinn,
            f_ext=loads,
            measured_disp=measured_values,
            measured_dofs=measured_dofs,
            config=config_pinn,
        )
        
        print(f"\n{'='*60}")
        print(f"RESULTADO:")
        print(f"{'='*60}")
        print(f"Convergió: {result_pinn.converged}")
        print(f"Iteraciones: {len(result_pinn.history)}")
        
        # ========================================
        # PASO 4: Comparar E(x)
        # ========================================
        print("\n4️⃣  Comparación E(x) identificado vs real:")
        
        x_eval = np.linspace(0, L, 50)
        E_real_vals = [young_modulus_real(x, E0, L) for x in x_eval]
        
        E_nn_vals = []
        for x in x_eval:
            E_nn = young_nn_property.value(x)
            if isinstance(E_nn, torch.Tensor):
                E_nn = E_nn.item()
            E_nn_vals.append(E_nn)
        
        print(f"\n   x [m]    E_real [Pa]    E_NN [Pa]    Error %")
        print(f"   " + "-" * 55)
        for x in [0.0, 1.0, 2.0, 3.0]:
            E_real = young_modulus_real(x, E0, L)
            E_nn = young_nn_property.value(x)
            if isinstance(E_nn, torch.Tensor):
                E_nn = E_nn.item()
            error = abs(E_nn - E_real) / E_real * 100
            print(f"   {x:4.1f}    {E_real:.3e}    {E_nn:.3e}    {error:6.2f}%")
        
        # ========================================
        # PASO 5: Graficar
        # ========================================
        print("\n5️⃣  Generando gráficos...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # E(x)
        ax1 = axes[0]
        ax1.plot(x_eval, np.array(E_real_vals)/1e11, 'b-', linewidth=2, label='E(x) real')
        ax1.plot(x_eval, np.array(E_nn_vals)/1e11, 'r--', linewidth=2, label='E(x) NN (NR)')
        ax1.scatter(nodes[[1,2]], [young_modulus_real(x, E0, L)/1e11 for x in nodes[[1,2]]], 
                   c='blue', s=100, marker='o', label='Mediciones', zorder=5)
        ax1.set_xlabel('Posición x [m]', fontsize=12)
        ax1.set_ylabel('Young modulus E(x) [×10¹¹ Pa]', fontsize=12)
        ax1.set_title('Newton-Raphson PINN', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Desplazamientos
        ax2 = axes[1]
        u_pinn = result_pinn.displacements.flatten()
        ax2.plot(nodes, u_measured * 1e6, 'bo-', linewidth=2, markersize=8, label='Mediciones')
        ax2.plot(nodes, u_pinn * 1e6, 'rs--', linewidth=2, markersize=8, label='PINN (NR)')
        ax2.set_xlabel('Nodo', fontsize=12)
        ax2.set_ylabel('Desplazamiento [μm]', fontsize=12)
        ax2.set_title('Desplazamientos', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pinn_newton_raphson.png', dpi=150, bbox_inches='tight')
        print(f"   ✅ Gráfico guardado: pinn_newton_raphson.png")
        plt.close()
        
    except Exception as e:
        print(f"\n❌ Error durante optimización PINN:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
