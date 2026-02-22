#!/usr/bin/env python3
"""
Generic solver for FEM and PINN problems
Reads problem.json and dispatches to appropriate solver

Usage:
    python generic.py problem.json

Format of problem.json:
{
  "nodes": [[x, y], ...],
  "elements": [[i, j], ...],
  "dirichlet": [true/false per node],  // Boundary conditions (fixed)
  "neumann": [[fx, fy], ...],          // Forces per node
  "material": {
    "young": 210e9,
    "area": 0.01,
    "density": 7850
  },
  "measured_displacements": {
    "nodes": [1, 2],                    // Which nodes have measurements
    "ux": [0.001, 0.002],               // Measured ux values
    "uy": [0.0, 0.0]                    // Measured uy values
  },
  "nn_properties": ["young"],           // Which properties to approximate with NN
  "nn_config": {
    "young": {"hidden_layers": 2, "neurons_per_layer": 20},
    "area": {"hidden_layers": 2, "neurons_per_layer": 20},
    "density": {"hidden_layers": 2, "neurons_per_layer": 20}
  },
  "pinn_config": {
    "max_iterations": 500,
    "learning_rate_u": 1e-7,
    "learning_rate_theta": 1e-4,
    "alpha_physics": 1.0,
    "alpha_data": 100.0
  },
  "solver_type": "fem" | "pinn-gd" | "pinn-nr"
}
"""

import sys
import json
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# Add FEM module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from fem.model import FEMModel, Material
from fem.properties import NNProperty
from fem.solver import solve, solve_gd, solve_nr, SolverConfig, SolverResult


# Global logger
logger = None


def setup_logging(problem_file):
    """Setup logging to file and console"""
    global logger

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    problem_name = Path(problem_file).stem
    log_file = Path(problem_file).parent / f"{problem_name}_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Log file created: {log_file}")
    logger.info(f"Problem file: {problem_file}")
    logger.info("=" * 60)

    return log_file


def log_print(msg="", level="info"):
    """Print to console and log file"""
    global logger
    if logger:
        if level == "debug":
            logger.debug(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
        else:
            logger.info(msg)
    else:
        print(msg)


class SimpleNN(nn.Module):
    """Generic NN for material properties"""

    def __init__(self, hidden_layers=2, neurons_per_layer=20):
        super().__init__()

        layers = []
        layers.append(nn.Linear(1, neurons_per_layer))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(neurons_per_layer, 1))

        self.net = nn.Sequential(*layers)

        # Initialize to output near 1.0 (before scaling)
        with torch.no_grad():
            self.net[-1].bias.fill_(1.0)
            self.net[-1].weight.fill_(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_problem(problem_file):
    """Parse problem.json into FEMModel and solver config"""

    log_print("\n[DEBUG] Starting parse_problem...", level="debug")

    with open(problem_file, "r") as f:
        data = json.load(f)

    # Parse nodes
    nodes_list = data.get("nodes", [])
    nodes = np.array([[n["x"], n["y"]] for n in nodes_list])
    n_nodes = len(nodes)
    n_dofs = n_nodes * 2

    log_print(f"[DEBUG] Nodes: {n_nodes}, DOFs: {n_dofs}", level="debug")
    log_print(f"[DEBUG] Node coordinates:\n{nodes}", level="debug")

    # Parse elements (support both formats: [[i,j], ...] or [{'nodes': [i,j]}, ...])
    elements_data = data.get("elements", [])
    if elements_data and isinstance(elements_data[0], list):
        # Simple format: [[i,j], ...]
        elements = np.array(elements_data)
    else:
        # Object format: [{'nodes': [i,j]}, ...]
        elements = np.array([[e["nodes"][0], e["nodes"][1]] for e in elements_data])

    log_print(f"[DEBUG] Elements: {len(elements)}", level="debug")
    log_print(f"[DEBUG] Element connectivity:\n{elements}", level="debug")

    # Parse boundary conditions (Dirichlet - fixed nodes)
    fixed_dofs = []
    for i, node in enumerate(nodes_list):
        if node.get("fixed", False):
            fixed_dofs.extend([2 * i, 2 * i + 1])  # Both x and y fixed
        else:
            if node.get("fixed_x", False):
                fixed_dofs.append(2 * i)
            if node.get("fixed_y", False):
                fixed_dofs.append(2 * i + 1)
    fixed_dofs = np.array(fixed_dofs, dtype=int)

    log_print(f"[DEBUG] Fixed DOFs: {fixed_dofs}", level="debug")

    # Parse loads (Neumann - forces)
    loads = data.get("loads", [0.0] * n_dofs)
    f_ext = np.array(loads)

    log_print(f"[DEBUG] External loads: {f_ext}", level="debug")

    # Parse material properties
    material_data = data.get("material", {})
    base_young = material_data.get("young", 210e9)
    base_area = material_data.get("area", 0.01)
    base_density = material_data.get("density", 7850)

    log_print(
        f"[DEBUG] Base material: E={base_young}, A={base_area}, ρ={base_density}",
        level="debug",
    )

    # Check if we need to create NN properties
    nn_config = data.get("nn_config", {})
    solver_type = data.get("solver_type", "fem")

    # Determine which properties use NN
    material_props = {}

    # Young modulus
    if nn_config.get("young", {}).get("enabled", False):
        young_arch = nn_config["young"]
        young_net = SimpleNN(
            hidden_layers=young_arch.get("hiddenLayers", 2),
            neurons_per_layer=young_arch.get("neuronsPerLayer", 20),
        )
        material_props["young"] = NNProperty(
            net=young_net, input_dim=1, enforce_positive=True, scale=base_young
        )
        log_print(f"[DEBUG] Young: NNProperty (scale={base_young})", level="debug")
    else:
        material_props["young"] = base_young
        log_print(f"[DEBUG] Young: Scalar ({base_young})", level="debug")

    # Area
    if nn_config.get("area", {}).get("enabled", False):
        area_arch = nn_config["area"]
        area_net = SimpleNN(
            hidden_layers=area_arch.get("hiddenLayers", 2),
            neurons_per_layer=area_arch.get("neuronsPerLayer", 20),
        )
        material_props["area"] = NNProperty(
            net=area_net, input_dim=1, enforce_positive=True, scale=base_area
        )
        log_print(f"[DEBUG] Area: NNProperty (scale={base_area})", level="debug")
    else:
        material_props["area"] = base_area
        log_print(f"[DEBUG] Area: Scalar ({base_area})", level="debug")

    # Density
    if nn_config.get("density", {}).get("enabled", False):
        density_arch = nn_config["density"]
        density_net = SimpleNN(
            hidden_layers=density_arch.get("hiddenLayers", 2),
            neurons_per_layer=density_arch.get("neuronsPerLayer", 20),
        )
        material_props["density"] = NNProperty(
            net=density_net, input_dim=1, enforce_positive=True, scale=base_density
        )
        log_print(f"[DEBUG] Density: NNProperty (scale={base_density})", level="debug")
    else:
        material_props["density"] = base_density
        log_print(f"[DEBUG] Density: Scalar ({base_density})", level="debug")

    # Create material
    material = Material(**material_props)
    log_print(f"[DEBUG] Material created successfully", level="debug")

    # Parse measured displacements for PINN
    measured_data = {}
    if solver_type.startswith("pinn"):
        measured_dofs = []
        measured_values = []

        for i, node in enumerate(nodes_list):
            ux_measured = node.get("measured_ux", 0)
            uy_measured = node.get("measured_uy", 0)

            # Only include non-zero measurements or explicitly flagged nodes
            if ux_measured != 0:
                measured_dofs.append(2 * i)
                measured_values.append(ux_measured)
            if uy_measured != 0:
                measured_dofs.append(2 * i + 1)
                measured_values.append(uy_measured)

        measured_data = {
            "dofs": np.array(measured_dofs, dtype=int),
            "values": np.array(measured_values),
        }

    # Create FEM model
    model = FEMModel(
        nodes=nodes,
        elements=elements,
        material=material,
        loads=f_ext,
        fixed_dofs=fixed_dofs,
        dimension=2,
    )

    # Unified solver configuration (combines FEM and PINN params)
    solver_config_data = data.get("solver_config", {})
    pinn_config_data = data.get("pinn_config", {})

    # DEBUG: Let's see what we actually get from JSON
    print(f"*** solver_config_data: {solver_config_data}")
    print(f"*** pinn_config_data: {pinn_config_data}")

    print("*** BEFORE METHOD SELECTION ***")  # Debug marker

    # Determine solver method (map old solver_type to new method)
    solver_type = data.get("solver_type", "auto")

    # Check if method is explicitly specified in solver_config first
    explicit_method = solver_config_data.get("method", None)

    print(f"[DEBUG] solver_type: {solver_type}, explicit_method: {explicit_method}")

    if explicit_method:
        method = explicit_method  # Use explicitly specified method
        print(f"[DEBUG] Using explicit method: {method}")
    elif solver_type == "fem":
        method = "nr"  # Classical FEM → Newton-Raphson (fallback)
        print(f"[DEBUG] Using FEM fallback method: {method}")
    elif solver_type in ["pinn-gd", "pinn"]:
        method = "gd"  # PINN → Gradient Descent (fallback)
        print(f"[DEBUG] Using PINN fallback method: {method}")
    else:
        method = "auto"  # Default fallback
        print(f"[DEBUG] Using auto fallback method: {method}")

    print(f"*** FINAL METHOD: {method} ***")  # Debug marker

    # Create unified SolverConfig with all parameters
    solver_config = SolverConfig(
        # Common
        max_iterations=pinn_config_data.get(
            "max_iterations", solver_config_data.get("max_iterations", 1000)
        ),
        tolerance=pinn_config_data.get(
            "tolerance", solver_config_data.get("tolerance", 1e-6)
        ),
        print_every=pinn_config_data.get("print_every", 10),
        # Newton-Raphson specific
        n_increments=solver_config_data.get("n_increments", 10),
        min_denominator=solver_config_data.get("min_denominator", 1e-10),
        # Gradient Descent specific
        learning_rate_u=pinn_config_data.get("learning_rate_u", 1e-7),
        learning_rate_theta=pinn_config_data.get("learning_rate_theta", 1e-4),
        # Loss weights
        alpha_physics=pinn_config_data.get("alpha_physics", 1.0),
        alpha_data=pinn_config_data.get("alpha_data", 100.0),
        # Method selection
        method=method,
    )

    log_print(
        f"[DEBUG] Solver config: method={solver_config.method}, tol={solver_config.tolerance}, max_iter={solver_config.max_iterations}",
        level="debug",
    )
    log_print(
        f"[DEBUG] Alpha weights: physics={solver_config.alpha_physics}, data={solver_config.alpha_data}",
        level="debug",
    )
    log_print(f"[DEBUG] parse_problem completed successfully", level="debug")

    return {
        "model": model,
        "solver_config": solver_config,
        "measured_data": measured_data,
    }


def solve_problem(parsed_data):
    """Execute appropriate solver using unified solver module"""

    model = parsed_data["model"]
    solver_config = parsed_data["solver_config"]
    measured_data = parsed_data.get("measured_data", {})

    log_print(f"\n{'='*60}")
    log_print(f"UNIFIED SOLVER")
    log_print(f"{'='*60}")
    log_print(f"Nodes: {len(model.nodes)}")
    log_print(f"Elements: {len(model.elements)}")
    log_print(f"Fixed DOFs: {len(model.fixed_dofs)}")
    log_print(f"Has NN: {model.material.has_trainable_params()}")
    log_print(f"Has measurements: {len(measured_data.get('dofs', [])) > 0}")
    log_print(f"Solver method: {solver_config.method}")

    # Extract measurements if present
    measured_disp = measured_data.get("values", None)
    measured_dofs = measured_data.get("dofs", None)

    # Call unified solver (auto-selects or uses specified method)
    result = solve(
        model=model,
        config=solver_config,
        measured_disp=measured_disp,
        measured_dofs=measured_dofs,
    )

    # Prepare output dictionary
    output = {
        "success": result.converged,
        "converged": result.converged,
        "iterations": len(result.history),
        "displacements": result.displacements.flatten().tolist(),
        "reactions": (
            result.reactions.flatten().tolist() if result.reactions is not None else []
        ),
        "history": result.history,
    }

    # Add NN properties if present
    if result.nn_parameters:
        output["nn_parameters"] = {
            k: v.tolist() for k, v in result.nn_parameters.items()
        }
        output["identified_properties"] = extract_nn_properties(model)

    return output


def extract_nn_properties(model):
    """Extract identified NN properties for output"""

    properties = {}

    # Sample properties at key positions
    x_samples = np.linspace(0, 3, 10)  # Sample along length

    material = model.material

    # Young modulus
    if hasattr(material.young, "net"):
        E_vals = []
        for x in x_samples:
            E = material.young.value(x)
            if isinstance(E, torch.Tensor):
                E = E.item()
            E_vals.append(float(E))
        properties["young"] = {"x": x_samples.tolist(), "values": E_vals, "type": "nn"}
    else:
        properties["young"] = {"value": float(material.young.value()), "type": "scalar"}

    # Area
    if hasattr(material.area, "net"):
        A_vals = []
        for x in x_samples:
            A = material.area.value(x)
            if isinstance(A, torch.Tensor):
                A = A.item()
            A_vals.append(float(A))
        properties["area"] = {"x": x_samples.tolist(), "values": A_vals, "type": "nn"}
    else:
        properties["area"] = {"value": float(material.area.value()), "type": "scalar"}

    # Density
    if hasattr(material.density, "net"):
        rho_vals = []
        for x in x_samples:
            rho = material.density.value(x)
            if isinstance(rho, torch.Tensor):
                rho = rho.item()
            rho_vals.append(float(rho))
        properties["density"] = {
            "x": x_samples.tolist(),
            "values": rho_vals,
            "type": "nn",
        }
    else:
        properties["density"] = {
            "value": float(material.density.value()),
            "type": "scalar",
        }

    return properties


def main():
    if len(sys.argv) < 2:
        print("Usage: python generic.py problem.json [output.json]")
        sys.exit(1)

    problem_file = sys.argv[1]

    # Setup logging
    log_file = setup_logging(problem_file)

    # Auto-generate output file if not provided
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        # Generate .res.json from problem.json
        problem_path = Path(problem_file)
        output_file = str(problem_path.parent / f"{problem_path.stem}.res.json")

    log_print(f"Output file will be: {output_file}")
    log_print("=" * 60)

    try:
        # Parse problem
        log_print("\n[STEP 1] Parsing problem file...")
        parsed_data = parse_problem(problem_file)
        log_print("✅ Problem parsed successfully")

        # Solve
        log_print("\n[STEP 2] Solving problem...")
        result = solve_problem(parsed_data)
        log_print("✅ Problem solved")

        # Output results
        log_print("\n[STEP 3] Writing results...")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        log_print(f"✅ Results written to {output_file}")

        # Also print summary
        log_print(f"\n{'='*60}")
        log_print("SOLUTION SUMMARY:")
        if result.get("success"):
            log_print(f"  Status: SUCCESS")
            if "iterations" in result:
                log_print(f"  Iterations: {result['iterations']}")
            if "displacements" in result:
                disp = result["displacements"]
                max_u = max([abs(d) for d in disp])
                log_print(f"  Max displacement: {max_u:.6e}")
        else:
            log_print(f"  Status: FAILED")
            if "error" in result:
                log_print(f"  Error: {result['error']}")

        log_print(f"{'='*60}")
        log_print("✅ Solve completed successfully")
        log_print(f"{'='*60}\n")
        log_print(f"Log file saved: {log_file}")

    except Exception as e:
        log_print(f"\n❌ Error: {e}", level="error")
        import traceback

        tb_str = traceback.format_exc()
        log_print(tb_str, level="error")
        sys.exit(1)


if __name__ == "__main__":
    main()
