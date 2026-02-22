# Python FEM (base port from MATLAB)

Primera base modular en Python para reproducir la lógica FEM existente:

- `fem/model.py`: estructuras del modelo y configuración de solver.
- `fem/geometry.py`: utilidades de geometría y mapeo de DOF.
- `fem/element.py`: **múltiples tipos de elementos** (1D lineal, 2D no lineal).
- `fem/assembly.py`: ensamblaje global de rigidez y fuerzas internas.
- `fem/boundary.py`: partición de grados de libertad libres/fijos.
- `fem/core.py`: solver incremental iterativo estilo Newton-Raphson.

## Ejemplos

### 1D Lineal (barra horizontal - recomendado para empezar)
Ejemplo simple con matriz de rigidez clásica K = (EA/L) * [1, -1; -1, 1]:

```bash
python -m examples.truss1d_simple
```

### 2D No lineal (geometría compleja tipo FEM_2D.m)

```bash
python -m examples.fem2d_like
```

## Tipos de elementos disponibles

1. **`truss1d_linear_element`**: Barra 1D, formulación lineal clásica
   - Matriz: K = (EA/L) * [1, -1; -1, 1]
   - Deformación infinitesimal
   - Converge en 1-2 iteraciones

2. **`truss2d_element_state`**: Truss 2D con no linealidad geométrica
   - Deformación Green-Lagrange  
   - Rigidez constitutiva + geométrica
   - Requiere múltiples iteraciones Newton-Raphson

Para elegir tipo: especificar `dimension=1` o `dimension=2` en `FEMModel`.
