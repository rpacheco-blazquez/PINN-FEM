"""Ejemplo con red neuronal: Young modulus variable E(x).

Demuestra el uso de NNProperty para aproximar propiedades materiales
que varían espacialmente.

Sistema:
    |--- o --- o --- o ---|
    ^                     ^
    
E(x) = NN(x)  ← Módulo de Young variable
"""

from __future__ import annotations

import sys
from pathlib import Path
# Add parent directory to path so we can import fem package
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np

from fem import FEMModel, Material, SolverConfig, solve_incremental_newton, NNProperty


class SimpleYoungNet(nn.Module):
    """Red neuronal simple para E(x)."""
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    print("=" * 60)
    print("Ejemplo FEM con NN: Young modulus E(x) variable")
    print("=" * 60)
    
    # Geometría 1D
    nodes = np.array([0.0, 1.0, 2.0, 3.0])
    elements = np.array([[0, 1], [1, 2], [2, 3]])
    
    # ============================================================
    # Opción 1: Young escalar constante (caso base)
    # ============================================================
    print("\n1️⃣  Caso BASE: Young constante E = 2.1e11 Pa")
    
    material_scalar = Material(
        young=2.1e11,    # ← Escalar (backward compatible)
        area=1e-4,
        density=7800.0,
    )
    
    model_scalar = FEMModel(
        nodes=nodes,
        elements=elements,
        material=material_scalar,
        loads=np.array([0, 0, 1000, 0]),
        fixed_dofs=[0, 3],
        dimension=1,
    )
    
    config = SolverConfig(n_increments=1, max_iterations=10, tolerance=1e-10)
    result_scalar = solve_incremental_newton(model_scalar, config)
    
    print(f"   Converged: {result_scalar.converged}")
    print(f"   Max displacement: {np.max(np.abs(result_scalar.displacements)):.6e} m")
    
    # ============================================================
    # Opción 2: Young con red neuronal E(x)
    # ============================================================
    print("\n2️⃣  Caso NN: Young variable E(x) = NN(x)")
    
    # Crear red neuronal
    young_net = SimpleYoungNet()
    
    # Inicializar pesos para que E(x) sea cercano a un valor constante
    with torch.no_grad():
        # Inicializar con valores pequeños para que la NN empiece casi constante
        for param in young_net.parameters():
            param.data.normal_(0, 0.01)
        # El último bias controla el valor base
        young_net.net[-1].bias.fill_(1.0)  # Pre-softplus
    
    # Crear NNProperty con escala explícita
    young_nn_property = NNProperty(
        net=young_net,
        input_dim=1,           # Depende de x (posición 1D)
        enforce_positive=True,  # softplus garantiza > 0
        scale=2.1e11,          # Escala a valores físicos correctos
    )
    
    # Verificar evaluación en diferentes posiciones
    print("\n   Evaluación de E(x) en diferentes puntos:")
    for x_test in [0.0, 1.0, 2.0, 3.0]:
        E_val = young_nn_property.value(x_test)
        print(f"      E({x_test:.1f}) = {E_val:.3e} Pa")
    
    material_nn = Material(
        young=young_nn_property,  # ← Red neuronal
        area=1e-4,
        density=7800.0,
    )
    
    print(f"\n   Material has trainable params: {material_nn.has_trainable_params()}")
    print(f"   Number of NN parameters: {sum(p.numel() for p in material_nn.get_all_torch_params())}")
    
    model_nn = FEMModel(
        nodes=nodes,
        elements=elements,
        material=material_nn,
        loads=np.array([0, 0, 1000, 0]),
        fixed_dofs=[0, 3],
        dimension=1,
    )
    
    result_nn = solve_incremental_newton(model_nn, config)
    
    print(f"\n   Converged: {result_nn.converged}")
    print(f"   Max displacement: {np.max(np.abs(result_nn.displacements)):.6e} m")
    
    # ============================================================
    # Comparación
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARACIÓN:")
    print("=" * 60)
    print(f"Desplazamientos nodales [m]:")
    print(f"  Nodo  |  Escalar      |  NN           |  Diff")
    print(f"  " + "-" * 50)
    for i in range(len(nodes)):
        u_scalar = result_scalar.displacements[i, 0]
        u_nn = result_nn.displacements[i, 0]
        diff = abs(u_scalar - u_nn)
        print(f"  {i}     |  {u_scalar:12.6e} | {u_nn:12.6e} | {diff:12.6e}")
    
    print("\n✅ Ambos métodos funcionan correctamente!")
    print("   - Escalar: sintaxis simple, rápida")
    print("   - NN: flexible, entrenable, puede variar espacialmente")


if __name__ == "__main__":
    main()
