#!/usr/bin/env python3
"""
API wrapper for PINN Newton-Raphson solver
Reads JSON input, runs PINN inverse problem solver with NR, writes JSON output

Usage:
    python api_pinn_newton_raphson.py input.json output.json
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

# Add FEM module to path
sys.path.insert(0, str(Path(__file__).parent))

from fem.nn_solver import pinn_inverse_problem_nr


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
    max_iterations = solver_config.get('max_iterations', 50)
    tolerance = solver_config.get('tolerance', 1e-6)
    lambda_lm = solver_config.get('lambda_lm', 1e-3)  # Levenberg-Marquardt damping
    
    return {
        'nodes': nodes,
        'elements': elements,
        'f_ext': f_ext,
        'fixed_dofs': fixed_dofs,
        'young_init': young_init,
        'area_init': area_init,
        'u_measured': u_measured,
        'measured_dofs': measured_dofs,
        'max_iterations': max_iterations,
        'tolerance': tolerance,
        'lambda_lm': lambda_lm,
        'n_dofs': n_dofs
    }


def solve_pinn_nr(problem):
    """Solve PINN inverse problem using Newton-Raphson (Gauss-Newton)"""
    
    print("Starting PINN Newton-Raphson solver...")
    print(f"  Measured DOFs: {len(problem['measured_dofs'])}")
    print(f"  Initial Young's modulus: {problem['young_init']:.3e} Pa")
    print(f"  Initial Area: {problem['area_init']:.6f} m²")
    print(f"  Max iterations: {problem['max_iterations']}")
    print(f"  Tolerance: {problem['tolerance']:.3e}")
    print(f"  LM damping: {problem['lambda_lm']:.3e}")
    
    result = pinn_inverse_problem_nr(
        nodes=problem['nodes'],
        elements=problem['elements'],
        f_ext=problem['f_ext'],
        fixed_dofs=problem['fixed_dofs'],
        young_init=problem['young_init'],
        area_init=problem['area_init'],
        u_measured=problem['u_measured'],
        measured_dofs=problem['measured_dofs'],
        max_iterations=problem['max_iterations'],
        tolerance=problem['tolerance'],
        lambda_lm=problem['lambda_lm']
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
    
    # Format convergence history
    convergence_history = [
        {
            'iteration': h['iteration'],
            'residual': h['residual'],
            'young': h['young'],
            'area': h['area']
        }
        for h in history
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
        'final_residual': float(history[-1]['residual']) if history else None
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python api_pinn_newton_raphson.py input.json output.json")
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
        result = solve_pinn_nr(problem)
        
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
