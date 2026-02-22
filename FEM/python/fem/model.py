from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np

from .properties import Property, to_property


@dataclass
class Material:
    """Material properties (can be scalar or NN-based).
    
    All properties are converted to Property objects internally.
    Access via: material.young.value(), material.area.value(x), etc.
    """
    young: Property | float
    area: Property | float
    density: Property | float = 0.0
    
    def __post_init__(self):
        # Auto-convert scalars to ScalarProperty
        self.young = to_property(self.young)
        self.area = to_property(self.area)
        self.density = to_property(self.density)
    
    def has_trainable_params(self) -> bool:
        """Check if any property has trainable parameters."""
        return (
            self.young.is_trainable() or 
            self.area.is_trainable() or 
            self.density.is_trainable()
        )
    
    def get_all_torch_params(self) -> list:
        """Get all trainable torch parameters from all properties."""
        params = []
        params.extend(self.young.get_torch_params())
        params.extend(self.area.get_torch_params())
        params.extend(self.density.get_torch_params())
        return params


@dataclass
class FEMModel:
    nodes: np.ndarray
    elements: np.ndarray
    material: Material
    loads: np.ndarray
    fixed_dofs: np.ndarray
    dimension: int = 2  # 1 for 1D truss, 2 for 2D truss

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.elements = np.asarray(self.elements, dtype=int)
        self.loads = np.asarray(self.loads, dtype=float).reshape(-1)
        self.fixed_dofs = np.asarray(self.fixed_dofs, dtype=int).reshape(-1)

        if self.dimension not in (1, 2):
            raise ValueError("dimension must be 1 or 2")

        if self.dimension == 1:
            if self.nodes.ndim != 1:
                raise ValueError("For 1D, nodes must be 1D array of positions")
        elif self.dimension == 2:
            if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
                raise ValueError("For 2D, nodes must have shape (nnode, 2)")

        if self.elements.ndim != 2 or self.elements.shape[1] != 2:
            raise ValueError("elements must have shape (nelm, 2)")

        ndof = self.ndof
        if self.loads.size != ndof:
            raise ValueError(f"loads size must be {ndof}, got {self.loads.size}")
        if np.any(self.fixed_dofs < 0) or np.any(self.fixed_dofs >= ndof):
            raise ValueError("fixed_dofs contain out-of-range indices")

    @property
    def nnode(self) -> int:
        if self.dimension == 1:
            return self.nodes.shape[0]
        return self.nodes.shape[0]

    @property
    def nelm(self) -> int:
        return self.elements.shape[0]

    @property
    def ndof(self) -> int:
        return self.nnode * self.dimension


@dataclass(frozen=True)
class SolverConfig:
    n_increments: int = 10
    max_iterations: int = 80
    tolerance: float = 1e-6
    min_denominator: float = 1e-12


@dataclass
class SolverResult:
    displacements: np.ndarray
    reactions: np.ndarray
    converged: bool
    history: List[Dict[str, float]] = field(default_factory=list)
