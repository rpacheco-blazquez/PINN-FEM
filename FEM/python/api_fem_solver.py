#!/usr/bin/env python3
"""
API wrapper for FEM classical solver
Reads JSON input, runs FEM solver, writes JSON output

Usage:
    python api_fem_solver.py input.json output.json
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add FEM module to path
sys.path.insert(0, str(Path(__file__).parent))

from fem.model import FEMModel, SolverConfig, Material
from fem.core import solve_incremental_newton


def parse_input(input_data):
    """Parse JSON input into FEMModel"""
    
    # Nodes: [[x, y], [x, y], ...]
    nodes = np.array([[n['x'], n['y']] for n in input_data['nodes']])
    n_nodes = len(nodes)
    n_dofs = n_nodes * 2
    
    # Elements: [[node_i, node_j], ...]
    elements = np.array([[e['nodes'][0], e['nodes'][1]] for e in input_data['elements']])
    
    # Material properties
    material_data = input_data.get('material', {})
    material = Material(
        young=material_data.get('young', 210e9),  # Pa
        area=material_data.get('area', 0.01),      # m²
        density=material_data.get('density', 7850)  # kg/m³
    )
    
    # Loads: force vector [fx0, fy0, fx1, fy1, ...]
    loads = input_data.get('loads', [0.0] * n_dofs)
    f_ext = np.array(loads)
    
    # Fixed DOFs from boundary conditions
    fixed_dofs = []
    for i, node in enumerate(input_data['nodes']):
        if node.get('fixed', False):
            fixed_dofs.extend([2*i, 2*i+1])  # Both x and y fixed
        elif node.get('fixed_x', False):
            fixed_dofs.append(2*i)
        elif node.get('fixed_y', False):
            fixed_dofs.append(2*i+1)
    
    fixed_dofs = np.array(fixed_dofs, dtype=int)
    
    # Solver configuration
    solver_config_data = input_data.get('solver_config', {})
    solver_config = SolverConfig(
        tolerance=solver_config_data.get('tolerance', 1e-6),
        max_iterations=solver_config_data.get('max_iterations', 50),
        n_increments=solver_config_data.get('n_increments', 10)
    )
    
    # Create FEM model
    model = FEMModel(
        nodes=nodes,
        elements=elements,
        material=material,
        loads=f_ext,
        fixed_dofs=fixed_dofs,
        dimension=2
    )
    
    return model, solver_config


def compute_element_stresses(model: FEMModel, u: np.ndarray):
    """Compute element stresses and strains"""
    
    nodes = model.nodes
    elements = model.elements
    young = model.material.young.value()
    
    stresses = []
    strains = []
    
    for elem in elements:
        i, j = elem
        
        # Node positions
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        
        # Displacements
        ui = u[2*i:2*i+2]
        uj = u[2*j:2*j+2]
        
        # Original length
        L0 = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        
        # Deformed length
        xi_def = xi + ui[0]
        yi_def = yi + ui[1]
        xj_def = xj + uj[0]
        yj_def = yj + uj[1]
        L = np.sqrt((xj_def - xi_def)**2 + (yj_def - yi_def)**2)
        
        # Strain (engineering)
        epsilon = (L - L0) / L0
        
        # Stress
        sigma = young * epsilon
        
        strains.append(float(epsilon))
        stresses.append(float(sigma))
    
    return stresses, strains


def main():
    if len(sys.argv) != 3:
        print("Usage: python api_fem_solver.py input.json output.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Reading input from {input_file}")
    
    try:
        # Read input
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Parse and solve
        model, solver_config = parse_input(input_data)
        
        print(f"Solving FEM problem:")
        print(f"  Nodes: {model.nnode}")
        print(f"  Elements: {model.nelm}")
        print(f"  DOFs: {model.ndof}")
        print(f"  Fixed DOFs: {len(model.fixed_dofs)}")
        print(f"  Increments: {solver_config.n_increments}")
        
        result = solve_incremental_newton(model, solver_config)
        
        # Compute stresses
        u_flat = result.displacements.reshape(-1)
        stresses, strains = compute_element_stresses(model, u_flat)
        
        # Format output
        output = {
            'displacements': u_flat.tolist(),
            'stresses': stresses,
            'strains': strains,
            'converged': result.converged,
            'convergence_history': result.history
        }
        
        # Write output
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"[OK] Results written to {output_file}")
        print(f"  Converged: {result.converged}")
        
    except Exception as e:
        # Write error to output
        error_output = {
            'error': str(e),
            'type': type(e).__name__
        }
        
        with open(output_file, 'w') as f:
            json.dump(error_output, f, indent=2)
        
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
