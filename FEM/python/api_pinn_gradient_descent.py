#!/usr/bin/env python3
"""
API wrapper for PINN Gradient Descent solver
Reads JSON input, runs PINN inverse problem solver, writes JSON output

Usage:
    python api_pinn_gradient_descent.py input.json output.json
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

# Add FEM module to path
sys.path.insert(0, str(Path(__file__).parent))

from fem.nn_solver_gd import pinn_inverse_problem_gd


def parse_input(input_data):
    """Parse JSON input into problem data"""
    
    # Nodes
    nodes = np.array([[n['x'], n['y']] for n in input_data['nodes']])
    n_nodes = len(nodes)
    n_dofs = n_nodes * 2
    
    # Elements
    elements = np.array([[e['nodes'][0], e['nodes'][1]] for e in input_data['elements']])
    
    # Material properties (initial guess)
    material = input_data.get('material', {})
    young_init = material.get('young', 210e9)
    area_init = material.get('area', 0.01)
    
    # Loads
    loads = input_data.get('loads', [0.0] * n_dofs)
    f_ext = np.array(loads)
    
    # Fixed DOFs
    fixed_dofs = []
    for i, node in enumerate(input_data['nodes']):
        if node.get('fixed', False):
            fixed_dofs.extend([2*i, 2*i+1])
        elif node.get('fixed_x', False):
            fixed_dofs.append(2*i)
        elif node.get('fixed_y', False):
            fixed_dofs.append(2*i+1)
    
    # Measured data (for inverse problem)
    measured_disp = input_data.get('measured_disp', [])
    measured_dofs = input_data.get('measured_dofs', [])
    
    if not measured_disp or not measured_dofs:
        raise ValueError("PINN requires measured_disp and measured_dofs for inverse problem")
    
    u_measured = np.array(measured_disp)
    measured_dofs = np.array(measured_dofs, dtype=int)
    
    # Solver configuration
    solver_config = input_data.get('solver_config', {})
    n_iterations = solver_config.get('max_iterations', 500)
    learning_rate = solver_config.get('learning_rate', 0.001)
    alpha = solver_config.get('alpha', 1.0)  # Physics weight
    beta = solver_config.get('beta', 100.0)   # Data weight
    
    # Parameter bounds (optional)
    young_bounds = solver_config.get('young_bounds', [1e9, 500e9])
    area_bounds = solver_config.get('area_bounds', [0.001, 0.1])
    
    return {
        'nodes': nodes,
        'elements': elements,
        'f_ext': f_ext,
        'fixed_dofs': fixed_dofs,
        'young_init': young_init,
        'area_init': area_init,
        'u_measured': u_measured,
        'measured_dofs': measured_dofs,
        'n_iterations': n_iterations,
        'learning_rate': learning_rate,
        'alpha': alpha,
        'beta': beta,
        'young_bounds': young_bounds,
        'area_bounds': area_bounds,
        'n_dofs': n_dofs
    }


def solve_pinn_gd(problem):
    """Solve PINN inverse problem using Gradient Descent"""
    
    print("Starting PINN Gradient Descent solver...")
    print(f"  Measured DOFs: {len(problem['measured_dofs'])}")
    print(f"  Initial Young's modulus: {problem['young_init']:.3e} Pa")
    print(f"  Initial Area: {problem['area_init']:.6f} m²")
    print(f"  Iterations: {problem['n_iterations']}")
    print(f"  Learning rate: {problem['learning_rate']}")
    
    result = pinn_inverse_problem_gd(
        nodes=problem['nodes'],
        elements=problem['elements'],
        f_ext=problem['f_ext'],
        fixed_dofs=problem['fixed_dofs'],
        young_init=problem['young_init'],
        area_init=problem['area_init'],
        u_measured=problem['u_measured'],
        measured_dofs=problem['measured_dofs'],
        n_iterations=problem['n_iterations'],
        learning_rate=problem['learning_rate'],
        alpha=problem['alpha'],
        beta=problem['beta']
    )
    
    # Extract results
    u_final = result['u_final']
    young_final = result['young_final']
    area_final = result['area_final']
    history = result['history']
    
    # Compute element stresses with identified parameters
    nodes = problem['nodes']
    elements = problem['elements']
    
    stresses = []
    strains = []
    
    for elem in elements:
        i, j = elem
        
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        
        ui = u_final[2*i:2*i+2]
        uj = u_final[2*j:2*j+2]
        
        L0 = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        
        xi_def = xi + ui[0]
        yi_def = yi + ui[1]
        xj_def = xj + uj[0]
        yj_def = yj + uj[1]
        L = np.sqrt((xj_def - xi_def)**2 + (yj_def - yi_def)**2)
        
        epsilon = (L - L0) / L0
        sigma = young_final * epsilon
        
        strains.append(float(epsilon))
        stresses.append(float(sigma))
    
    # Format training history
    convergence_history = [
        {
            'iteration': h['iteration'],
            'loss_total': h['loss_total'],
            'loss_physics': h['loss_physics'],
            'loss_data': h['loss_data'],
            'young': h['young'],
            'area': h['area']
        }
        for h in history[::10]  # Save every 10th iteration
    ]
    
    return {
        'displacements': u_final.tolist(),
        'stresses': stresses,
        'strains': strains,
        'identified_params': {
            'young': float(young_final),
            'area': float(area_final)
        },
        'convergence_history': convergence_history,
        'final_loss': float(history[-1]['loss_total']) if history else None
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python api_pinn_gradient_descent.py input.json output.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Reading input from {input_file}")
    
    try:
        # Read input
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Parse and solve
        problem = parse_input(input_data)
        result = solve_pinn_gd(problem)
        
        # Write output
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"[OK] Results written to {output_file}")
        print(f"  Identified Young's modulus: {result['identified_params']['young']:.3e} Pa")
        print(f"  Identified Area: {result['identified_params']['area']:.6f} m^2")
        
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
