#!/usr/bin/env python
"""Test unitario para truss2d_linear_element_torch"""

import numpy as np
import torch
import sys

sys.path.insert(0, ".")

from fem.element import truss2d_linear_element
from fem.nn_assembly import truss2d_linear_element_torch


def test_single_element_horizontal():
    """Test: 1 barra horizontal, E=1, A=1, L=1"""
    print("=" * 70)
    print("TEST 1: Barra horizontal simple")
    print("=" * 70)

    # Geometría
    x_i0 = np.array([0.0, 0.0])
    x_j0 = np.array([1.0, 0.0])

    # Desplazamientos de prueba
    u_i_np = np.array([0.0, 0.0])
    u_j_np = np.array([1.0, 0.0])  # Tracción pura en X

    u_i_torch = torch.tensor(u_i_np, dtype=torch.float32, requires_grad=True)
    u_j_torch = torch.tensor(u_j_np, dtype=torch.float32, requires_grad=True)

    # Propiedades
    E = 1.0
    A = 1.0

    # NumPy version
    ke_np, fe_np = truss2d_linear_element(x_i0, x_j0, u_i_np, u_j_np, E, A)

    # Torch version
    E_torch = torch.tensor(E, dtype=torch.float32)
    A_torch = torch.tensor(A, dtype=torch.float32)
    ke_torch, fe_torch = truss2d_linear_element_torch(
        x_i0, x_j0, u_i_torch, u_j_torch, E_torch, A_torch
    )

    print("\n1. Matriz de rigidez:")
    print("   NumPy:")
    print(ke_np)
    print("\n   Torch:")
    print(ke_torch.detach().numpy())

    diff_ke = np.abs(ke_np - ke_torch.detach().numpy()).max()
    print(f"\n   Max diff: {diff_ke:.2e}", "✓ PASS" if diff_ke < 1e-6 else "✗ FAIL")

    print("\n2. Fuerza interna:")
    print(f"   NumPy: {fe_np}")
    print(f"   Torch: {fe_torch.detach().numpy()}")

    diff_fe = np.abs(fe_np - fe_torch.detach().numpy()).max()
    print(f"\n   Max diff: {diff_fe:.2e}", "✓ PASS" if diff_fe < 1e-6 else "✗ FAIL")

    # Test gradientes
    print("\n3. Gradientes (autograd):")
    loss = torch.sum(fe_torch)
    loss.backward()

    print(
        f"   ∂loss/∂u_i = {u_i_torch.grad.numpy() if u_i_torch.grad is not None else 'None'}"
    )
    print(
        f"   ∂loss/∂u_j = {u_j_torch.grad.numpy() if u_j_torch.grad is not None else 'None'}"
    )

    has_grad = u_i_torch.grad is not None and u_j_torch.grad is not None
    print(f"\n   Gradients computed: {'✓ PASS' if has_grad else '✗ FAIL'}")

    return diff_ke < 1e-6 and diff_fe < 1e-6 and has_grad


def test_three_bars_assembly():
    """Test: 3 barras en serie (como example1)"""
    print("\n" + "=" * 70)
    print("TEST 2: 3 barras en serie (assembly)")
    print("=" * 70)

    # Nodos
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    # Elementos
    elements = np.array([[0, 1], [1, 2], [2, 3]])

    # CASO A: Desplazamiento inicial incorrecto (u=0)
    print("\n--- CASO A: u = 0 (inicial) ---")
    u_initial = torch.zeros(8, dtype=torch.float32, requires_grad=True)

    E = 1.0
    A = 1.0

    # Ensamblar fuerzas internas
    f_int_A = torch.zeros(8, dtype=torch.float32)

    for elem in elements:
        node_i, node_j = elem
        dofs = [2 * node_i, 2 * node_i + 1, 2 * node_j, 2 * node_j + 1]

        x_i0 = nodes[node_i]
        x_j0 = nodes[node_j]
        u_i = u_initial[[2 * node_i, 2 * node_i + 1]]
        u_j = u_initial[[2 * node_j, 2 * node_j + 1]]

        E_torch = torch.tensor(E, dtype=torch.float32)
        A_torch = torch.tensor(A, dtype=torch.float32)

        ke, fe = truss2d_linear_element_torch(x_i0, x_j0, u_i, u_j, E_torch, A_torch)

        for i_local, i_global in enumerate(dofs):
            f_int_A[i_global] += fe[i_local]

    print(f"   f_int = {f_int_A.detach().numpy()}")

    # Fuerza externa
    f_ext = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    print(f"   f_ext = {f_ext.numpy()}")

    # Residuo en DOFs libres
    free_dofs = [2, 4, 6]
    residual_A = f_int_A[free_dofs] - f_ext[free_dofs]

    print(f"   R[free] = {residual_A.detach().numpy()}")
    print(f"   ||R|| = {torch.norm(residual_A).item():.2e}")

    # Gradiente
    loss_A = torch.mean(residual_A**2)
    loss_A.backward()

    print(f"   ∂loss/∂u = {u_initial.grad.numpy()}")
    print(f"   ||grad|| = {torch.norm(u_initial.grad).item():.2e}")

    # CASO B: Desplazamientos correctos
    print("\n--- CASO B: u = solución correcta ---")
    u_correct = torch.tensor(
        [
            0.0,
            0.0,  # Node 0 fixed
            1.0,
            0.0,  # Node 1
            2.0,
            0.0,  # Node 2
            3.0,
            0.0,  # Node 3
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    f_int_B = torch.zeros(8, dtype=torch.float32)

    for elem in elements:
        node_i, node_j = elem
        dofs = [2 * node_i, 2 * node_i + 1, 2 * node_j, 2 * node_j + 1]

        x_i0 = nodes[node_i]
        x_j0 = nodes[node_j]
        u_i = u_correct[[2 * node_i, 2 * node_i + 1]]
        u_j = u_correct[[2 * node_j, 2 * node_j + 1]]

        E_torch = torch.tensor(E, dtype=torch.float32)
        A_torch = torch.tensor(A, dtype=torch.float32)

        ke, fe = truss2d_linear_element_torch(x_i0, x_j0, u_i, u_j, E_torch, A_torch)

        for i_local, i_global in enumerate(dofs):
            f_int_B[i_global] += fe[i_local]

    print(f"   f_int = {f_int_B.detach().numpy()}")
    print(f"   f_ext = {f_ext.numpy()}")

    residual_B = f_int_B[free_dofs] - f_ext[free_dofs]
    print(f"   R[free] = {residual_B.detach().numpy()}")
    print(f"   ||R|| = {torch.norm(residual_B).item():.2e}")

    loss_B = torch.mean(residual_B**2)
    loss_B.backward()

    print(f"   ∂loss/∂u = {u_correct.grad.numpy()}")
    print(f"   ||grad|| = {torch.norm(u_correct.grad).item():.2e}")

    return torch.norm(residual_B).item() < 1e-6


def test_diagonal_bar():
    """Test: Barra diagonal a 45 grados"""
    print("\n" + "=" * 70)
    print("TEST 3: Barra diagonal 45°")
    print("=" * 70)

    # Geometría: barra de (0,0) a (1,1)
    x_i0 = np.array([0.0, 0.0])
    x_j0 = np.array([1.0, 1.0])
    L = np.sqrt(2)

    # Desplazamiento: nodo j se mueve 0.1 en dirección de la barra
    # Dirección unitaria: (1/√2, 1/√2)
    displacement = 0.1
    u_i_np = np.array([0.0, 0.0])
    u_j_np = np.array([displacement / np.sqrt(2), displacement / np.sqrt(2)])

    u_i_torch = torch.tensor(u_i_np, dtype=torch.float32)
    u_j_torch = torch.tensor(u_j_np, dtype=torch.float32, requires_grad=True)

    E = 100.0
    A = 1.0

    # NumPy version
    ke_np, fe_np = truss2d_linear_element(x_i0, x_j0, u_i_np, u_j_np, E, A)

    # Torch version
    E_torch = torch.tensor(E, dtype=torch.float32)
    A_torch = torch.tensor(A, dtype=torch.float32)
    ke_torch, fe_torch = truss2d_linear_element_torch(
        x_i0, x_j0, u_i_torch, u_j_torch, E_torch, A_torch
    )

    print(f"\n1. Geometría:")
    print(f"   L = {L:.4f}")
    print(f"   cx = cy = {1/np.sqrt(2):.4f}")

    print(f"\n2. Desplazamiento nodo j:")
    print(f"   u_j = {u_j_np}")
    print(f"   Alargamiento axial = {displacement:.4f}")

    print(f"\n3. Fuerza interna:")
    print(f"   NumPy: {fe_np}")
    print(f"   Torch: {fe_torch.detach().numpy()}")

    # Fuerza axial esperada: F = (EA/L) * δL = (100*1/√2) * 0.1
    F_expected = (E * A / L) * displacement
    print(f"\n   Fuerza axial esperada: {F_expected:.4f}")
    print(f"   Fuerza en nodo j (x): {fe_torch[2].item():.4f}")
    print(f"   Fuerza en nodo j (y): {fe_torch[3].item():.4f}")

    diff_fe = np.abs(fe_np - fe_torch.detach().numpy()).max()
    print(f"\n   Max diff: {diff_fe:.2e}", "✓ PASS" if diff_fe < 1e-5 else "✗ FAIL")

    return diff_fe < 1e-5


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" TESTS UNITARIOS: truss2d_linear_element_torch")
    print("=" * 70)

    results = []

    try:
        results.append(("Barra horizontal", test_single_element_horizontal()))
    except Exception as e:
        print(f"\n✗ FAIL: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Barra horizontal", False))

    try:
        results.append(("Assembly 3 barras", test_three_bars_assembly()))
    except Exception as e:
        print(f"\n✗ FAIL: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Assembly 3 barras", False))

    try:
        results.append(("Barra diagonal", test_diagonal_bar()))
    except Exception as e:
        print(f"\n✗ FAIL: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Barra diagonal", False))

    # Resumen
    print("\n" + "=" * 70)
    print(" RESUMEN")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:.<50} {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 70)

    sys.exit(0 if passed == total else 1)
