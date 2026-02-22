# Examples - JSON Input Format

Este directorio contiene ejemplos de problemas FEM resueltos con diferentes métodos numéricos.

## 📁 Archivos Principales

### `example1.json` - Newton-Raphson (FEM Clásico)
**Descripción:** Problema FEM forward con solver Newton-Raphson.

**Configuración:**
- **Geometría:** 3 barras horizontales en serie (4 nodos en x=0,1,2,3)
- **Material:** E=1.0, A=1.0, ρ=1.0
- **Condiciones de contorno:** Nodo 0 fijo, nodos 1-3 solo libres en dirección x
- **Carga:** F=1.0 N aplicada en nodo 3 (dirección x)
- **Solver:** `"solver_type": "fem"` (Newton-Raphson)
- **Parámetros:** max_iterations=50, tolerance=1e-6

**Solución teórica:**
```
u = [0, 1, 2, 3] metros
```

**Resultados:**
- Iteraciones: **10**
- Convergencia: ✓ Sí
- Precisión: ~1e-16 (precisión de máquina)

---

### `example2.json` - Gradient Descent (PINN-GD)
**Descripción:** Mismo problema que Example 1 pero resuelto con Gradient Descent.

**Configuración:**
- **Geometría:** Idéntica a Example 1
- **Material:** Idéntico a Example 1
- **Solver:** `"solver_type": "pinn-gd"` (Gradient Descent con Adam)
- **Parámetros:**
  ```json
  "solver_config": {
    "max_iterations": 10000,
    "tolerance": 1e-6,
    "learning_rate_u": 0.01,
    "alpha_residual": 1.0,
    "print_every": 1000
  }
  ```

**Resultados:**
- Iteraciones: **~3035**
- Convergencia: ✓ Sí (residual < 1e-6)
- Precisión: ~1e-6 (error < 4 micrones)

---

### `example3.json` - PINN Inverse Problem (NN Material Properties)
**Descripción:** Problema inverso donde una red neuronal aprende E(x,y) a partir de mediciones de desplazamientos.

**Configuración:**
- **Geometría:** Idéntica a Examples 1 & 2
- **Material:** Young's modulus representado por NN(x,y) con 2 capas ocultas de 20 neuronas
- **Measurements:** Desplazamientos medidos en nodos 1, 2, 3: [1.0, 2.0, 3.0] m
- **Solver:** `"solver_type": "pinn-gd"` (Gradient Descent con Adam)
- **Loss function:** 
  ```
  L = α_physics * ||R||² + α_data * ||u_measured - u||²
  ```
- **Parámetros:**
  ```json
  "pinn_config": {
    "learning_rate_u": 0.01,
    "learning_rate_theta": 0.001,
    "alpha_physics": 1.0,
    "alpha_data": 100.0,
    "max_iterations": 5000,
    "tolerance": 1e-6
  }
  ```

**Arquitectura de NN:**
```python
Input: (x, y) → Hidden(20) → Tanh → Hidden(20) → Tanh → Output(1)
Young = softplus(NN(x,y)) * scale  # Garantiza E > 0
```

**Resultados:**
- Iteraciones: **1110**
- Convergencia: ✓ Sí (loss=9.85e-07)
- Final loss_physics: 6.798e-07 (excellent equilibrium)
- Final loss_data: 1.976e-07 (fits measurements perfectly)
- NN parameters: 501 trainable params (20×2 + 20 + 20×20 + 20 + 1×20 + 1)

**Desplazamientos finales vs medidos:**
| Nodo | Measured | Predicted | Error (%) |
|------|----------|-----------|-----------|
| 1    | 1.0000   | 0.9998    | 0.02%     |
| 2    | 2.0000   | 2.0000    | 0.002%    |
| 3    | 3.0000   | 2.9998    | 0.008%    |

**Conclusión:** El PINN aprende correctamente E(x,y) que satisface tanto las ecuaciones de física como los datos medidos.

---

### `example4.json` - PINN con TODAS las Propiedades como NN  
**Descripción:** Problema inverso avanzado donde múltiples redes neuronales aprenden E(x,y), A(x,y) y ρ(x,y) simultáneamente.

**Configuración:**
- **Geometría:** Idéntica a Examples 1, 2 & 3
- **Material:** Todas las propiedades representadas por NNs independientes
- **Measurements:** Mismos desplazamientos medidos: [1.0, 2.0, 3.0] m
- **Solver:** `"solver_type": "pinn-gd"` (Gradient Descent con Adam)
- **Arquitecturas de NN:**
  - Young: NN(x,y) → 2×20×20×1 = 501 params
  - Area: NN(x,y) → 2×15×15×1 = 316 params  
  - Density: NN(x,y) → 2×10×10×1 = 141 params
  - **Total: 958 parámetros entrenables**

**Parámetros:**
```json
"nn_config": {
  "young": {"enabled": true, "input_dim": 2, "neurons_per_layer": 20},
  "area": {"enabled": true, "input_dim": 2, "neurons_per_layer": 15},
  "density": {"enabled": true, "input_dim": 2, "neurons_per_layer": 10}
},
"pinn_config": {
  "learning_rate_theta": 0.0005  // Más lento para estabilidad
}
```

**Resultados:**
- Iteraciones: **2684** (más que example3 debido a más parámetros)
- Convergencia: ✓ Sí (loss=9.97e-07)
- NN parameters: ~24.26 (vs 8.9 en example3)

**Predicciones de las 3 NNs (centroides de elementos):**
| Elemento | Young (E) | Area (A) | Density (ρ) | **E×A** |
|----------|-----------|----------|-------------|---------|
| 0        | 0.998     | 1.002    | 1.269       | **1.000** |
| 1        | 0.966     | 1.034    | 1.291       | **0.999** |
| 2        | 0.939     | 1.065    | 1.306       | **1.000** |

**Análisis de Coherencia:**
- ✅ **Compensación inteligente:** E↓ mientras A↑ para mantener rigidez EA≈1.0
- ✅ **Convergencia cooperativa:** 3 NNs trabajando juntas en lugar de competir
- ✅ **Física respetada:** ρ no afecta la estática, solo participa marginalmente

**Conclusión:** Las múltiples NNs aprenden relaciones físicas complejas distribuyendo roles cooperativamente para satisfacer equilibrio + datos.

---

## 📊 Comparación de Resultados

### Tabla de Ejemplos
| Example | Solver | Material | Iterations | NN Params | Purpose |
|---------|--------|----------|------------|-----------|---------|
| example1.json | Newton-Raphson | E=1.0 (constante) | 10 | 0 | Forward problem (clásico FEM) |
| example2.json | Gradient Descent | E=1.0 (constante) | 3035 | 0 | Forward problem (PINN-GD) |
| example3.json | Gradient Descent | E=NN(x,y) | 1110 | 501 | Inverse problem (learn E from data) |
| example4.json | Gradient Descent | E,A,ρ=NN(x,y) | 2684 | 958 | Multi-property inverse problem |

### Desplazamientos Nodales(Examples 1 & 2)
| Nodo | Example1 (NR) | Example2 (GD) | Diferencia |
|------|---------------|---------------|------------|
| 0    | 0.000000      | 0.000000      | 0.00e+00   |
| 1    | 1.000000      | 0.999998      | 1.79e-06   |
| 2    | 2.000000      | 1.999997      | 2.98e-06   |
| 3    | 3.000000      | 2.999996      | 3.58e-06   |

### Eficiencia Computacional
| Método            | Iteraciones | Ratio vs NR | NN Params |
|-------------------|-------------|-------------|-----------|
| Newton-Raphson (E=const) | 10     | 1x          | 0         |
| Gradient Descent (E=const) | 3035 | **303x**    | 0         |
| Gradient Descent (E=NN, inverse) | 1110 | **111x** | 501       |
| Gradient Descent (E,A,ρ=NN, multi) | 2684 | **268x** | 958       |

**Conclusión:** Para problemas FEM lineales forward, Newton-Raphson es ~300x más eficiente. Para problemas inversos con NN, GD es la única opción (NR no puede optimizar propiedades). La complejidad crece sub-linealmente con el número de NNs.

---

## 🚀 Cómo Ejecutar

### Ejecutar Example 1 (Newton-Raphson)
```bash
cd FEM/python
python examples/json/generic.py examples/json/example1.json
```

### Ejecutar Example 2 (Gradient Descent)
```bash
cd FEM/python
python examples/json/generic.py examples/json/example2.json
```

### Ejecutar Example 3 (PINN Inverse Problem)
```bash
cd FEM/python
python examples/json/generic.py examples/json/example3.json
```

### Ejecutar Example 4 (PINN Multi-Property)  
```bash
cd FEM/python
python examples/json/generic.py examples/json/example4.json
```

### Archivos de Salida
Cada ejecución genera:
- `exampleX.res.json` - Resultados (desplazamientos, reacciones, convergencia, historial de loss)
- `exampleX.log` - Log detallado de la ejecución

---

## 📝 Notas Importantes

### Cuándo usar Newton-Raphson (Example 1)
✓ Problemas FEM forward lineales  
✓ Materiales con propiedades conocidas  
✓ Máxima eficiencia computacional  
✓ Convergencia cuadrática (muy rápida)  

### Cuándo usar Gradient Descent sin NN (Example 2)
✓ Validación de implementación PINN-GD  
✓ Comparación con Newton-Raphson  
✓ Debugging de solver GD  

### Cuándo usar PINN con NN (Examples 3 & 4)
✓ **Problemas inversos:** identificar propiedades materiales desde mediciones  
✓ **Material heterogéneo:** E(x,y,z) varía espacialmente  
✓ **Data-driven modeling:** aprender constitutive laws desde experimentos  
✓ **Physics-informed learning:** combinar ecuaciones físicas + datos  
✓ **Multi-property identification:** identificar múltiples propiedades simultáneamente (Example 4)  

### Parámetros Críticos para GD
⚠️ **IMPORTANTE:** El learning rate debe configurarse correctamente:
```json
"pinn_config": {
  "learning_rate_u": 0.01,      // Learning rate para desplazamientos
  "learning_rate_theta": 0.001  // Learning rate para parámetros de NN
}
```

Con `lr_u=1e-7` (default), GD tarda ~1,000,000 iteraciones. Con `lr_u=0.01`, converge en ~3000.

### Estructura de Loss para PINN (Example 3)
```python
# Loss total
L = α_physics * L_physics + α_data * L_data

# Physics loss (equilibrio)
L_physics = 0.5 * ||R||² = 0.5 * ||f_internal - f_external||²

# Data loss (ajuste a mediciones)
L_data = ||u_measured - u_predicted||²

# Pesos recomendados
α_physics = 1.0    # Siempre > 0 (garantiza equilibrio)
α_data = 100.0     # Mayor peso → mejor ajuste a datos
```

---

## 🔍 Archivos Adicionales

### `example1-1.json` / `example2-2.json`
Casos de prueba con **1 solo elemento** para debugging y validación:
- `example1-1.json`: 1 elemento, Newton-Raphson
- `example2-2.json`: 1 elemento, Gradient Descent (~352 iteraciones)

### `generic.py`
Parser y ejecutor principal que:
1. Lee el archivo JSON de entrada
2. Construye el modelo FEM
3. Invoca el solver apropiado
4. Escribe resultados en formato JSON

---

## 📖 Formato JSON de Entrada

### Formato Básico (Examples 1 & 2)
```json
{
  "description": "Descripción del problema",
  "nodes": [
    {"x": 0.0, "y": 0.0, "fixed_x": true, "fixed_y": true}
  ],
  "elements": [[0, 1], [1, 2]],
  "loads": [0.0, 0.0, 1.0, 0.0],
  "material": {
    "young": 1.0,
    "area": 1.0,
    "density": 1.0
  },
  "solver_type": "fem" | "pinn-gd",
  "solver_config": {
    "max_iterations": 50,
    "tolerance": 1e-6,
    "learning_rate_u": 0.01  // Solo para pinn-gd
  }
}
```

### Formato Avanzado con PINN (Examples 3 & 4)
```json
{
  "description": "PINN inverse problem",
  "nodes": [...],
  "elements": [[0, 1], [1, 2]],
  "loads": [...],
  "material": {
    "young": 1.0,    // Usado como scale factor para NN
    "area": 1.0,     // Si NN enabled, usado como scale
    "density": 1.0   // Si NN enabled, usado como scale
  },
  "nn_config": {
    "young": {
      "enabled": true,
      "input_dim": 2,            // 1=(x), 2=(x,y), 3=(x,y,z)
      "hidden_layers": 2,
      "neurons_per_layer": 20
    },
    "area": {                    // Example 4: Multiple NNs
      "enabled": true,
      "input_dim": 2,
      "hidden_layers": 2,
      "neurons_per_layer": 15
    },
    "density": {                 // Example 4: Multiple NNs
      "enabled": true,
      "input_dim": 2,
      "hidden_layers": 2,
      "neurons_per_layer": 10
    }
  },
  "measured_displacements": {
    "nodes": [1, 2, 3],          // Node IDs con mediciones
    "ux": [1.0, 2.0, 3.0],       // Despl. en x
    "uy": [0.0, 0.0, 0.0]        // Despl. en y
  },
  "solver_type": "pinn-gd",
  "pinn_config": {
    "learning_rate_u": 0.01,
    "learning_rate_theta": 0.001,
    "alpha_physics": 1.0,
    "alpha_data": 100.0,
    "max_iterations": 5000,
    "tolerance": 1e-6,
    "print_every": 100
  }
}
```

**Campos clave para PINN:**
- `nn_config`: Define qué propiedades usan NN y su arquitectura
- `measured_displacements`: Datos experimentales para problem inverse
- `pinn_config`: Hiperparámetros de training (learning rates, alpha weights)

Para más detalles ver `generic.py`.
