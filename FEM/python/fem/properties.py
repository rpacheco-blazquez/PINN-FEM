"""Propiedades flexibles: escalares o redes neuronales.

Cualquier propiedad material (young, density, area, etc.) puede ser:
- ScalarProperty: valor constante
- NNProperty: red neuronal que depende de inputs espaciales o estado

Todos se acceden mediante .value(inputs=None)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class Property:
    """Interfaz base para propiedades materiales."""

    def value(self, inputs: Optional[Any] = None) -> float | np.ndarray:
        """Evalúa la propiedad.

        Args:
            inputs: Argumentos opcionales (posición, estado, etc.)
                   - None para propiedades constantes
                   - float/array para propiedades que dependen de posición
                   - dict para propiedades que dependen de múltiples variables

        Returns:
            Valor escalar o array de la propiedad
        """
        raise NotImplementedError

    def is_trainable(self) -> bool:
        """True si la propiedad tiene parámetros entrenables."""
        return False

    def get_torch_params(self) -> list:
        """Retorna lista de parámetros PyTorch para optimización."""
        return []


class ScalarProperty(Property):
    """Propiedad con valor escalar constante."""

    def __init__(self, value: float):
        self._value = float(value)

    def value(self, inputs: Optional[Any] = None) -> float:
        """Retorna el valor constante (ignora inputs)."""
        return self._value

    def __repr__(self) -> str:
        return f"ScalarProperty({self._value:.3e})"


class NNProperty(Property):
    """Propiedad aproximada por red neuronal.

    La NN puede depender de:
    - Posición espacial: x, (x,y), (x,y,z)
    - Estado del material: strain, stress, temperature, etc.
    - Cualquier combinación
    """

    def __init__(
        self,
        net: Any,  # torch.nn.Module
        input_dim: int = 1,
        enforce_positive: bool = True,
        scale: float = 1.0,
    ):
        """
        Args:
            net: Red neuronal (torch.nn.Module)
            input_dim: Dimensión del input (1 para x, 2 para (x,y), etc.)
            enforce_positive: Si True, aplica softplus para asegurar valores positivos
            scale: Factor de escala para la salida de la red
        """
        self.net = net
        self.input_dim = input_dim
        self.enforce_positive = enforce_positive
        self.scale = scale

        # Cache para evitar reimportar torch cada vez
        self._torch = None

    @property
    def torch(self):
        """Lazy import de torch."""
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def value(self, inputs: Optional[Any] = None) -> float | np.ndarray:
        """Evalúa la NN.

        Args:
            inputs:
                - None: usa input cero (propiedad "constante" pero entrenable)
                - float: posición 1D
                - array: posiciones múltiples o input multidimensional
                - dict: {'x': ..., 'strain': ..., etc.}

        Returns:
            Valor(es) de la propiedad evaluada
        """
        torch = self.torch

        # Preparar input
        if inputs is None:
            # Sin inputs espaciales: NN aprende un valor "constante"
            x_torch = torch.zeros(1, self.input_dim, dtype=torch.float32)
        elif isinstance(inputs, dict):
            # Inputs nombrados: concatenar en orden
            x_list = []
            for key in sorted(inputs.keys()):
                val = inputs[key]
                if isinstance(val, (int, float)):
                    x_list.append([val])
                else:
                    x_list.append(np.atleast_1d(val))
            x_torch = torch.tensor(np.column_stack(x_list), dtype=torch.float32)
        else:
            # Input directo (escalar o array)
            x_np = np.atleast_1d(inputs).astype(float)

            # Si es 1D y coincide con input_dim, es UN punto con múltiples coordenadas
            if x_np.ndim == 1:
                if len(x_np) == self.input_dim:
                    # Un punto con input_dim coordenadas: reshape a (1, input_dim)
                    x_np = x_np.reshape(1, -1)
                else:
                    # Múltiples puntos 1D: reshape a (n_points, 1)
                    x_np = x_np.reshape(-1, 1)

            if x_np.shape[1] < self.input_dim:
                # Pad con ceros si falta dimensión
                pad = np.zeros((x_np.shape[0], self.input_dim - x_np.shape[1]))
                x_np = np.column_stack([x_np, pad])
            x_torch = torch.tensor(x_np, dtype=torch.float32)

        # Forward pass
        # Only use no_grad if we're not in a gradient-tracking context
        # This allows the NN to be trained when used with gradient descent
        if self.torch.is_grad_enabled():
            # Gradient-enabled mode (for training) - return torch tensor
            output = self.net(x_torch)

            # Aplicar restricciones
            if self.enforce_positive:
                output = torch.nn.functional.softplus(output)

            output = output * self.scale

            # Return torch tensor (keep gradients)
            if isinstance(inputs, (int, float)) or inputs is None:
                return output.squeeze()
            return output
        else:
            # Regular evaluation mode (no gradients needed) - return numpy
            with torch.no_grad():
                output = self.net(x_torch)

                # Aplicar restricciones
                if self.enforce_positive:
                    output = torch.nn.functional.softplus(output)

                output = output * self.scale

                # Convertir a numpy
                result = output.squeeze().numpy()

                # Si era input escalar, retornar escalar
                if isinstance(inputs, (int, float)) or inputs is None:
                    return float(result) if result.size == 1 else result
                return result

    def is_trainable(self) -> bool:
        return True

    def get_torch_params(self) -> list:
        return list(self.net.parameters())

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.net.parameters())
        return f"NNProperty(dim={self.input_dim}, params={n_params}, scale={self.scale:.3e})"


def to_property(value: Any) -> Property:
    """Convierte un valor a Property si no lo es ya.

    Args:
        value: float, int, o Property

    Returns:
        Property object
    """
    if isinstance(value, Property):
        return value
    if isinstance(value, (int, float)):
        return ScalarProperty(float(value))
    raise TypeError(f"Cannot convert {type(value)} to Property")
