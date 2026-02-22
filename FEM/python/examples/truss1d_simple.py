"""Ejemplo simple: Barra 1D horizontal con carga axial.

Sistema:
    |--- o --- o --- o ---|
    ^   1     2     3     ^
    
- Nodo 0: fijo (x=0)
- Nodo 1: libre (x=1)
- Nodo 2: libre (x=2)  
- Nodo 3: fijo (x=3)
- Carga: F=1000 N en nodo 2 (hacia derecha)

Rigidez lineal: K = (EA/L) * [1, -1; -1, 1]
"""

from __future__ import annotations

import numpy as np

from fem import FEMModel, Material, SolverConfig, solve_incremental_newton


def main() -> None:
    # Geometría 1D (posiciones nodales en x)
    nodes = np.array([0.0, 1.0, 2.0, 3.0])
    
    # Conectividad (elementos barra)
    elements = np.array([
        [0, 1],  # elemento 0: nodo 0 -> nodo 1
        [1, 2],  # elemento 1: nodo 1 -> nodo 2
        [2, 3],  # elemento 2: nodo 2 -> nodo 3
    ])
    
    # Material (acero típico)
    material = Material(
        young=2.1e11,   # Pa (210 GPa)
        area=1e-4,      # m² (1 cm²)
        density=7800.0,
    )
    
    # Cargas (axiales)
    ndof = nodes.shape[0]  # 4 nodos = 4 DOF en 1D
    loads = np.zeros(ndof)
    loads[2] = 1000.0  # 1000 N en nodo 2
    
    # Condiciones de contorno (extremos fijos)
    fixed_dofs = np.array([0, 3])  # nodos 0 y 3 fijos
    
    # Crear modelo 1D
    model = FEMModel(
        nodes=nodes,
        elements=elements,
        material=material,
        loads=loads,
        fixed_dofs=fixed_dofs,
        dimension=1,  # ← CLAVE: dimension=1 para formulación 1D lineal
    )
    
    # Configuración solver (para lineal, 1 incremento es suficiente)
    config = SolverConfig(
        n_increments=1,
        max_iterations=10,
        tolerance=1e-10,
    )
    
    # Resolver
    result = solve_incremental_newton(model, config)
    
    # Resultados
    print("=" * 60)
    print("Resultado FEM 1D - Barra con carga axial")
    print("=" * 60)
    print(f"Converged: {result.converged}")
    print(f"\nDesplazamientos nodales [m]:")
    for i, u in enumerate(result.displacements.flatten()):
        print(f"  Nodo {i}: u = {u:12.6e}")
    
    print(f"\nReacciones en soportes [N]:")
    print(f"  Nodo 0 (fijo): R = {result.reactions[0, 0]:12.3f}")
    print(f"  Nodo 3 (fijo): R = {result.reactions[3, 0]:12.3f}")
    
    # Verificación analítica
    # Para barra con carga concentrada en medio:
    # Reacción izquierda = -F * (L_derecha / L_total)
    # Reacción derecha = -F * (L_izquierda / L_total)
    print(f"\nCheck equilibrio: sum(R) + F = {result.reactions[0, 0] + result.reactions[3, 0] + loads[2]:.3e}")
    
    print("\nHistorial de convergencia:")
    for step in result.history:
        print(f"  Inc {int(step['increment'])}: iter={int(step['iterations'])}, "
              f"res={step['residual']:.3e}, strain_max={step['max_strain']:.3e}")


if __name__ == "__main__":
    main()
