"""Ejemplo de problema inverso PINN: identificar E(x) desde mediciones.

Problema:
    1. Sistema real: barra con E(x) variable (conocido)
    2. "Medimos" desplazamientos en algunos nodos
    3. Intentamos recuperar E(x) usando NN + Newton-Raphson

    |--- o --- o --- o ---|
    ^                     ^
    
    E_real(x) = E0 * (1 + 0.5*x/L)  ← función lineal
    E_nn(x) = NN(x)                 ← a identificar
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
    NNProperty, solve_pinn_gradient_descent, PINNGradientDescentConfig
)


class YoungModulusNet(nn.Module):
    """Red neuronal para E(x)."""
    
    def __init__(self, hidden_size: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        
        # Inicialización: queremos salida cercana a 1.0 (pre-softplus)
        with torch.no_grad():
            self.net[-1].bias.fill_(1.0)
            self.net[-1].weight.fill_(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1) -> out: (batch, 1)"""
        return self.net(x)


def young_modulus_real(x: float, E0: float = 2.1e11, L: float = 3.0) -> float:
    """E(x) real (lineal): E = E0 * (1 + 0.5*x/L)"""
    return E0 * (1.0 + 0.5 * x / L)


def main():
    print("=" * 60)
    print("PROBLEMA INVERSO PINN: Identificar E(x)")
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
    
    # Crear modelo con E(x) real usando valores escalares por elemento
    # (aproximación: E constante en cada elemento)
    model_real = FEMModel(
        nodes=nodes,
        elements=elements,
        material=Material(
            young=E0,  # Usaremos valor promedio temporalmente
            area=A,
            density=7800.0,
        ),
        fixed_dofs=fixed_dofs,
        loads=loads,
        dimension=1,
    )
    
    # Resolver con solver estándar para obtener "mediciones"
    # NOTA: Aquí simplificamos usando E constante para generar datos sintéticos
    # En un caso real, tendrías mediciones experimentales
    result_real = solve_incremental_newton(model_real, SolverConfig())
    u_measured = result_real.displacements.flatten()  # "Mediciones"
    
    print(f"   Desplazamientos 'medidos':")
    for i, u in enumerate(u_measured):
        print(f"      Nodo {i}: u = {u:.6e} m")
    
    # ========================================
    # PASO 2: Definir modelo PINN con E(x) = NN(x)
    # ========================================
    print("\n2️⃣  Creando modelo PINN con E(x) = NN(x)...")
    
    young_net = YoungModulusNet(hidden_size=20)
    young_nn_property = NNProperty(
        net=young_net,
        input_dim=1,
        enforce_positive=True,
        scale=E0,  # Escalar salida a orden de magnitud correcto
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
    print(f"      Arquitectura: 1 → 20 → 20 → 1")
    
    # ========================================
    # PASO 3: Resolver problema inverso con Gradient Descent PINN
    # ========================================
    print("\n3️⃣  Resolviendo problema inverso con Gradient Descent PINN...")
    print("   (Identificando E(x) desde mediciones...)\n")
    
    # DOFs donde tenemos mediciones (nodos 1 y 2, excluimos nodos fijos 0,3)
    measured_dofs = [1, 2]
    measured_values = u_measured[measured_dofs]
    
    config_pinn = PINNGradientDescentConfig(
        max_iterations=500,
        tolerance=1e-7,
        learning_rate_u=1e-7,     # Learning rate para desplazamientos
        learning_rate_theta=1e-4,  # Learning rate para parámetros NN
        alpha_physics=1.0,         # Peso de ecuación de equilibrio
        alpha_data=100.0,          # Peso de ajuste a mediciones
        print_every=50,
    )
    
    try:
        result_pinn = solve_pinn_gradient_descent(
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
        # PASO 4: Comparar E(x) identificado vs real
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
        
        # Imprimir en puntos clave
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
        # PASO 5: Graficar resultados
        # ========================================
        print("\n5️⃣  Generando gráficos...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Subplot 1: E(x) real vs identificado
        ax1 = axes[0]
        ax1.plot(x_eval, np.array(E_real_vals)/1e11, 'b-', linewidth=2, label='E(x) real')
        ax1.plot(x_eval, np.array(E_nn_vals)/1e11, 'r--', linewidth=2, label='E(x) NN identificado')
        ax1.scatter(nodes[[1,2]], [young_modulus_real(x, E0, L)/1e11 for x in nodes[[1,2]]], 
                   c='blue', s=100, marker='o', label='Nodos con mediciones', zorder=5)
        ax1.set_xlabel('Posición x [m]', fontsize=12)
        ax1.set_ylabel('Young modulus E(x) [×10¹¹ Pa]', fontsize=12)
        ax1.set_title('Identificación de E(x)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Desplazamientos comparados
        ax2 = axes[1]
        u_pinn = result_pinn.displacements.flatten()
        ax2.plot(nodes, u_measured * 1e6, 'bo-', linewidth=2, markersize=8, label='Mediciones')
        ax2.plot(nodes, u_pinn * 1e6, 'rs--', linewidth=2, markersize=8, label='PINN solution')
        ax2.set_xlabel('Nodo', fontsize=12)
        ax2.set_ylabel('Desplazamiento [μm]', fontsize=12)
        ax2.set_title('Desplazamientos: Medidos vs PINN', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pinn_inverse_problem.png', dpi=150, bbox_inches='tight')
        print(f"   ✅ Gráfico guardado: pinn_inverse_problem.png")
        plt.close()  # Cerrar en lugar de mostrar interactivamente
        
    except Exception as e:
        print(f"\n❌ Error durante optimización PINN:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
