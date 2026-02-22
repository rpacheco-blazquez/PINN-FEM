# FEM Python - Documentación Técnica

## Estructura de módulos

### `fem/model.py`
Define las estructuras de datos principales:
- **`Material`**: propiedades (Young, área, densidad)
- **`FEMModel`**: geometría completa (nodos, elementos, material, cargas, BC)
- **`SolverConfig`**: parámetros del solver (incrementos, iteraciones, tolerancia)
- **`SolverResult`**: resultados (desplazamientos, reacciones, convergencia, historial)

### `fem/geometry.py`
Utilidades geométricas:
- **`element_dofs(i, j)`**: devuelve índices globales de DOF para elemento `i-j`
- **`split_element_data(...)`**: extrae coordenadas iniciales y desplazamientos de nodos

### `fem/element.py` ⭐
**Núcleo de formulación elemental** - Múltiples tipos soportados:

---

#### 1️⃣ Elemento 1D Lineal: `truss1d_linear_element(x_i0, x_j0, u_i, u_j, young, area)`

**Formulación clásica de barra 1D** (pequeños desplazamientos):

**Matriz de rigidez:**
```python
K_elem = (EA/L) * [1, -1; -1, 1]
```
- **Constitutiva únicamente** (no hay rigidez geométrica)
- Independiente del estado de deformación
- Simétrica y siempre positiva definida

**Deformación lineal:**
```python
epsilon = (u_j - u_i) / L
```
- Medida infinitesimal (válida solo para pequeños desplazamientos)

**Fuerza interna:**
```python
f_int = (EA/L) * [u_i - u_j, u_j - u_i] = K_elem * u_elem
```
- Consistente con f_int = K * u
- Para sistemas lineales: converge en 1-2 iteraciones

**Cuándo usar:**
- Análisis lineal estático
- Problemas con pequeños desplazamientos
- Como base para PINN (formulación simple)
- Testing y validación rápida

---

#### 2️⃣ Elemento 2D No Lineal: `truss2d_element_state(x_i0, x_j0, u_i, u_j, young, area)`

**Truss 2D con no linealidad geométrica** (grandes desplazamientos):

**Retorna:** `ElementState(ke_total, fe_int, strain)`

**Componentes de rigidez:

1. **Rigidez constitutiva (lineal material)**  
   ```python
   ke_l = (E*A / L0³) * d0 ⊗ d0
   ```
   - Basada en geometría **inicial** (no deformada)
   - Representa comportamiento **material lineal**
   - Independiente del estado de deformación

2. **Rigidez geométrica (no lineal)**  
   ```python
   ke_nl = (E*A / L0) * e_gl * d ⊗ d
   ```
   - Basada en geometría **deformada actual**
   - Proporcional a la deformación `e_gl` (Green-Lagrange)
   - Captura **efectos de segundo orden** (P-Delta, endurecimiento/ablandamiento)

3. **Rigidez tangente total**
   ```python
   ke_total = ke_l + ke_nl
   ```
   - Usada en Newton-Raphson para resolver incrementalmente

#### Deformación Green-Lagrange:
```python
e_gl = (L² - L0²) / (2*L0²)
```
- `L0`: longitud inicial
- `L`: longitud deformada actual
- Medida de deformación **finita** (válida para grandes desplazamientos)

#### Fuerza interna:
```python
fe_int = (E*A / L0) * e_gl * d
```
- Fuerza nodal equivalente del elemento en configuración deformada

**Cuándo usar:**
- Análisis no lineal geométrico
- Grandes desplazamientos
- Replicar comportamiento de MATLAB original
- Validación de convergencia Newton-Raphson

---

### `fem/assembly.py`
Ensamblaje global con **despacho automático** según dimensión:
- **`dimension=1`**: loop usa `truss1d_linear_element()`
- **`dimension=2`**: loop usa `truss2d_element_state()`

Para cada elemento: extrae geometría → calcula `ke_total` y `fe_int` → ensambla en `K_global` y `F_int`

Retorna: `(K_tangent, F_internal, max_strain)`

---

### `fem/boundary.py`
Manejo de condiciones de contorno:
- **`free_and_fixed_dofs(ndof, fixed_dofs)`**: particiona DOF en libres y restringidos

### `fem/core.py`
Solver incremental Newton-Raphson:
- **`solve_incremental_newton(model, config)`**:
  1. Loop de incrementos de carga (load stepping)
  2. Para cada incremento: iteraciones Newton-Raphson
     - Ensambla sistema tangente
     - Resuelve: `K_tan * dU = F_ext - F_int`
     - Actualiza desplazamientos: `U += dU`
     - Verifica convergencia: `||dU|| / ||U|| < tol`
  3. Calcula reacciones finales
  4. Retorna `SolverResult`

---

## Flujo de ejecución típico

```python
from fem import FEMModel, Material, SolverConfig, solve_incremental_newton
import numpy as np

# 1. Definir geometría
nodes = np.array([[0, 0], [1, 0], [2, 0]])
elements = np.array([[0, 1], [1, 2]])

# 2. Material y cargas
material = Material(young=2.1e11, area=1e-2)
loads = np.zeros(nodes.shape[0] * 2)
loads[4] = 1000.0  # Fx en nodo 2

# 3. BC (fijos: nodo 0 completo, nodo 1 y_blocked)
fixed_dofs = np.array([0, 1, 3, 5])

# 4. Crear modelo
model = FEMModel(nodes, elements, material, loads, fixed_dofs)

# 5. Resolver
config = SolverConfig(n_increments=10, max_iterations=50, tolerance=1e-6)
result = solve_incremental_newton(model, config)

# 6. Resultados
print(f"Converged: {result.converged}")
print(f"Displacements:\n{result.displacements}")
print(f"Reactions:\n{result.reactions}")
```

---

## Formulación implementada

### Elementos disponibles:
- ✅ **Truss 1D lineal** (pequeños desplazamientos, formulación clásica)
- ✅ **Truss 2D no lineal** (grandes desplazamientos, Green-Lagrange)
- ✅ **Material lineal elástico** (Hooke)
- ✅ **Newton-Raphson incremental** (tangent stiffness method)

### No implementado (aún):
- ❌ Elementos 3D
- ❌ Elementos viga (flexión)
- ❌ Material no lineal (plasticidad, daño)
- ❌ Dinámica (masa, amortiguamiento)
- ❌ Contacto

---

## Uso de formulación lineal vs no lineal

**Ya implementado**: Para usar formulación **puramente lineal**, simplemente especifica `dimension=1`:

```python
# Análisis lineal 1D (recomendado para PINN)
model = FEMModel(
    nodes=np.array([0.0, 1.0, 2.0]),  # posiciones 1D
    elements=[[0, 1], [1, 2]],
    material=Material(young=2.1e11, area=1e-4),
    loads=np.array([0, 0, 1000]),
    fixed_dofs=[0, 2],
    dimension=1,  # ← Formulación lineal: K = (EA/L)*[1,-1;-1,1]
)
```

Para análisis **no lineal 2D**, usa `dimension=2`:

```python
# Análisis no lineal 2D (grandes desplazamientos)
model = FEMModel(
    nodes=np.array([[0,0], [1,0], [2,0]]),  # posiciones (x,y)
    elements=[[0, 1], [1, 2]],
    material=Material(young=2.1e11, area=1e-4),
    loads=np.zeros(6),
    fixed_dofs=[0, 1, 4, 5],
    dimension=2,  # ← Formulación no lineal: ke_l + ke_nl
)
```

---

## Extensión PINN (próximo paso)

Para añadir **Physics-Informed Neural Network** en el loop iterativo:

1. **Reemplazar ensamblaje** por evaluación de red neuronal
2. **Loss function** combina:
   - Residuo físico (equilibrio FEM)
   - Condiciones de contorno
   - Datos de medición (si existen)
3. **PyTorch backend** en lugar de NumPy para autodiff

Estructura actual ya está lista para esta transición (modular y separada).
