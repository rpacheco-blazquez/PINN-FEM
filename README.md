# PINN-FEM: Physics-Informed Neural Networks + Finite Element Method

Sistema completo de elementos finitos con identificación de parámetros usando redes neuronales informadas por la física.

## Arquitectura

```
┌─────────────┐      HTTP/REST      ┌─────────────┐      spawn      ┌─────────────┐
│   Frontend  │ ──────────────────> │   Backend   │ ─────────────> │   Python    │
│ React+Vite  │                     │ Node/Express│                │  FEM/PINN   │
│  Port 3001  │ <────────────────── │  Port 5000  │ <───────────── │   Solvers   │
└─────────────┘     JSON Results    └─────────────┘   JSON I/O     └─────────────┘
```

### Componentes

**Frontend (React + Vite)**
- Canvas interactivo para crear/editar mallado FEM
- Tabla tipo Excel con coordenadas, conectividades y resultados
- Panel de propiedades de materiales y condiciones de frontera
- Botón de cálculo con selección de solver (FEM clásico, PINN-GD, PINN-NR)
- Visualización de deformaciones con escala ajustable

**Backend (Node.js + Express)**
- API REST para invocar solvers de Python
- Gestión de archivos temporales para comunicación con Python
- Rutas: `/api/fem/solve` (FEM clásico), `/api/fem/solve-pinn` (problema inverso)
- Timeout y manejo de errores

**Python Solvers**
- `api_fem_solver.py`: Solver clásico con Newton-Raphson incremental
- `api_pinn_gradient_descent.py`: Identificación de parámetros con gradient descent (PyTorch)
- `api_pinn_newton_raphson.py`: Identificación con Gauss-Newton + Levenberg-Marquardt

## Instalación

### Prerequisitos
- Node.js >= 18
- Python >= 3.8
- pip

### Instalación Rápida (Recomendada)

Desde la raíz del proyecto:

```bash
npm run install:all
```

Esto instalará las dependencias del root, backend y frontend.

### Instalación Manual

Si prefieres instalar por separado:

**Backend (Node.js)**
```bash
cd backend
npm install
```

**Frontend (React)**
```bash
cd frontend
npm install
```

**Root (para scripts de desarrollo)**
```bash
npm install
```

### Python Dependencies

```bash
cd FEM/python
pip install numpy torch matplotlib
```

O con el virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install numpy torch matplotlib
```

## Ejecución

### Opción 1: Levantar Frontend y Backend Simultáneamente (Recomendado)

Desde la raíz del proyecto:

```bash
npm run dev
```

Esto levantará:
- Backend en `http://localhost:5000` con nodemon (auto-reload)
- Frontend en `http://localhost:3001` con Vite (hot-reload)

### Opción 2: Levantar por Separado

**Backend:**
```bash
npm run dev:backend
# o
cd backend && npm run dev
```

**Frontend:**
```bash
npm run dev:frontend
# o
cd frontend && npm run dev
```

### Abrir en el navegador

Navega a `http://localhost:3001` y verás la interfaz CAD.

## Uso

### Crear Modelo FEM

1. **Añadir Nodos**: Haz clic en el canvas para crear nodos
2. **Crear Elementos**: Modo "Element", haz clic en dos nodos para conectarlos
3. **Condiciones de Frontera**: En la tabla "Nodes", marca la casilla "Fixed" para nodos empotrados
4. **Propiedades**: Modulo de Young, área, densidad (hardcoded por ahora en `App.jsx`)

### Ejecutar Cálculo

1. Selecciona el tipo de solver:
   - **FEM Clásico**: Solver estándar de elementos finitos
   - **PINN (Gradient Descent)**: Identificación de parámetros con optimización iterativa
   - **PINN (Newton-Raphson)**: Identificación más rápida con método de segundo orden

2. Configura tolerancia y máximo de iteraciones

3. Haz clic en "🚀 Calcular FEM"

### Visualizar Resultados

- Los desplazamientos se muestran en el canvas con interpolación de forma
- Escala de deformación ajustable (1x - 1000x)
- Tabla "Results" muestra Ux, Uy y magnitud |U| por nodo
- Para PINN, se muestran los parámetros identificados (Young, Area)

## Ejemplos y Benchmarks

El sistema incluye 6 ejemplos de validación que demuestran diferentes capacidades del solver unificado con carga incremental y **sistema de warm start optimizado**.

### Configuración de Ejemplos

Todos los ejemplos utilizan:
- **Geometría**: Barra 1D de 4 nodos (3 elementos)
- **Carga incremental**: 10 incrementos de 0% a 100%
- **Sistema warm start**: Inicialización inteligente desde incremento anterior
- **Tolerancia**: 1×10⁻⁶

### Tabla de Performance Comparativa

| **Ejemplo** | **Solver** | **Preconditioning** | **Iteraciones** | **Tiempo** | **Comentarios** |
|-------------|------------|---------------------|-----------------|------------|-----------------|
| **Example 1** | Newton-Raphson | N/A | **10** | ~1s | Óptimo directo |
| **Example 2** | Gradient Descent | ❌ No | **~2,500** | ~5.6s | Baseline GD |
| **Example 2-P** | Gradient Descent | ✅ Sí | **~1,290** | ~3.1s | **45% más rápido** |
| **Example 3** | PINN+GD | ❌ No | **~2,200** | ~13s | E = NN(x,y,λ) |
| **Example 3-P** | PINN+GD | ✅ Sí | **~2,638** | ~9s | **31% más rápido** |
| **Example 4** | PINN+GD | ❌ No | **~3,500** | ~3min | E,A,ρ = NNs |
| **Example 4-P** | PINN+GD | ✅ Sí | **~2,126** | ~18s | **90% más rápido** |
| **Example 5** | Híbrido | ❌ No | **~20** | ~0.67s | GD→NR sin precon |
| **Example 5-P** | Híbrido | ✅ Sí | **~900** | ~2.4s | GD→NR con precon |
| **Example 6** | Híbrido+NN | ❌ No | **2000** | ~7.6s | **❌ FAILED** |
| **Example 6-P** | Híbrido+NN | ✅ Sí | **~900** | ~7.0s | **✅ SUCCESS** |
| **Example 7** | Híbrido+3NNs | ❌ No | **~79** | ~24.2s | **✅ SUCCESS** |
| **Example 7-P** | Híbrido+3NNs | ✅ Sí | **~1,236** | **~10.5s** | **🚀 56% más rápido** |
| **Example 8** | Full Newton-Raphson | N/A | **10** | ~0.6s | Verifica full-nr ≡ nr |
| **Example 9** | Full NR+NN | N/A | **~1000** | ~60s | E=NN, Hessiano costoso |
| **Example 10** | Full NR+3NNs | N/A | **~1000** | >120s | E,A,ρ=NNs, Hessiano 837×837 |

### Detalles por Ejemplo

#### Example 1: Newton-Raphson Clásico
```bash
cd FEM/python/examples/json && python generic.py example1.json
```
- **Solver**: Newton-Raphson directo
- **Material**: Propiedades escalares constantes (E=A=ρ=1.0)
- **Performance**: 1 iteración por incremento (solver directo)
- **Uso**: Validación de convergencia y referencia de performance

#### Example 2: Gradient Descent Puro

**Base Example (sin preconditioning):**
```bash
cd FEM/python/examples/json && python generic.py example2.json
```
- **Solver**: Gradient Descent sin redes neurales
- **Material**: Propiedades escalares constantes (E=A=ρ=1.0)
- **Preconditioning**: ❌ Deshabilitado (`"preconditioning": false`)
- **Performance**: 
  - Incremento 1 (❄️ cold start): ~237 iterations
  - Incrementos 2-10 (🔥 warm start): ~250-270 iterations
  - **Total**: ~2,500 iteraciones, ~5.6 segundos

**Variante con Preconditioning:**
```bash
cd FEM/python/examples/json && python generic.py example2-P.json
```
- **Solver**: Gradient Descent con preconditioning habilitado
- **Material**: Igual (propiedades escalares constantes)
- **Preconditioning**: ✅ Habilitado (`"preconditioning": true`)
- **Performance**:
  - Fase preconditioning: ~80 iteraciones (tolerancia relajada 1e-4)
  - Fase principal: ~42 iteraciones (tolerancia estricta 1e-6)
  - **Total**: ~1,290 iteraciones, ~3.1 segundos
  - **🚀 beneficio**: 45% reducción de tiempo, 48% menos iteraciones

**Uso**: Comparación de convergencia GD con y sin preconditioning

#### Example 3: PINN con Young NN
```bash
cd FEM/python/examples/json && python generic.py example3.json
```
- **Solver**: PINN + Gradient Descent
- **Material**: E = NN(x,y,λ), A,ρ = scalar
- **NN Architecture**: 2 capas × 20 neuronas, input_dim=3
- **Performance**: 
  - Incremento 1: ~1,200 iterations (NN learning + equilibrium)
  - Incrementos 2-10: ~100-120 iterations (warm start)
- **Uso**: Identificación de módulo de Young variable espacialmente y con carga

#### Example 3-P: PINN con Young NN + Preconditioning

**Variante con Preconditioning:**
```bash
cd FEM/python/examples/json && python generic.py example3-P.json
```
- **Solver**: PINN + Gradient Descent con preconditioning habilitado
- **Material**: Igual (E = NN(x,y,λ), A,ρ = scalar)
- **Preconditioning**: ✅ Habilitado (`"preconditioning": true`)
- **Performance**:
  - Incremento 1 (cold start): ~1,707 iteraciones (NN learning + precon + final)
  - Incrementos 2-10 (warm start): ~84-196 iteraciones cada uno
  - **Total**: ~2,638 iteraciones, ~9 segundos
  - **🚀 beneficio**: 31% reducción de tiempo vs Example 3
- **Uso**: Alternativa más eficiente para identificación PINN con 1 NN

#### Example 4: PINN Multi-Propiedad
```bash
cd FEM/python/examples/json && python generic.py example4.json
```
- **Solver**: PINN + Gradient Descent
- **Material**: 3 NNs independientes:
  - Young: NN(x,y,λ) - 2×20 neuronas
  - Area: NN(x,y,λ) - 2×15 neuronas  
  - Density: NN(x,y,λ) - 2×10 neuronas
- **Performance**: 
  - Incremento 1: ~2,755 iterations (3 NNs learning simultaneously)
  - Incrementos 2-10: ~80-150 iterations (warm start + trained NNs)
- **Uso**: Caso más complejo - identificación simultánea de múltiples propiedades

#### Example 5: Solver Híbrido (GD + Newton-Raphson)

**Base Example (sin preconditioning):**
```bash
cd FEM/python/examples/json && python generic.py example5.json
```
- **Solver**: Híbrido (GD preconditioning + NR finalization)
- **Material**: Propiedades escalares constantes (sin NNs)
- **Preconditioning**: ❌ Deshabilitado (`"preconditioning": false`)
- **Estrategia**: Saltar Fase 1 (GD) → Fase 2 directa (NR)
- **Performance**:
  - Por incremento: 2 iteraciones NR solamente
  - **Total**: ~20 iteraciones, ~0.67 segundos
  - **🚀 óptimo**: Comportamiento similar a Example 1 (NR directo)

**Variante con Preconditioning:**
```bash
cd FEM/python/examples/json && python generic.py example5-P.json
```
- **Solver**: Híbrido completo (GD + NR)
- **Material**: Igual (propiedades escalares constantes)
- **Preconditioning**: ✅ Habilitado (`"preconditioning": true`)
- **Estrategia**: Fase 1 (GD ~80 iter) → Fase 2 (NR ~2 iter)
- **Performance**:
  - **Total**: ~900 iteraciones, ~2.4 segundos
  - **Insight**: Para problemas lineales, el preconditioning es innecesario

**Uso**: Demostrar el comportamiento del solver híbrido y cuándo usar preconditioning

#### Example 6: PINN Híbrido con Neural Networks

**Base Example (sin preconditioning):**
```bash
cd FEM/python/examples/json && python generic.py example6.json
```
- **Solver**: Híbrido (GD → GD finalization para NNs)
- **Material**: E = NN(x,y,λ), con A,ρ escalares
- **Measured Data**: Desplazamientos objetivo en nodos [1,2,3]
- **Preconditioning**: ❌ Deshabilitado (`"preconditioning": false`)
- **Performance**:
  - **❌ FRACASO TOTAL**: No converge en 2000 iteraciones
  - Loss final: 6.578e-06 (no alcanza tolerancia 1e-06)
  - Solo completa 1 de 10 incrementos
  - **Tiempo**: ~7.6 segundos (desperdiciados)

**Variante con Preconditioning:**
```bash 
cd FEM/python/examples/json && python generic.py example6-P.json
```
- **Solver**: Híbrido completo con preconditioning habilitado
- **Material**: Igual (E = NN, A,ρ escalares)
- **Preconditioning**: ✅ Habilitado (`"preconditioning": true`)
- **Estrategia**: Fase 1 (GD precon ~300 iter) → Fase 2 (GD main ~581 iter)
- **Performance**:
  - **✅ ÉXITO COMPLETO**: Converge los 10 incrementos
  - Incremento 1: 881 iteraciones totales
  - Incrementos 2-10: ~90 iteraciones c/u (warm start)
  - **Total**: ~900 iteraciones, ~7.0 segundos
  - **🎯 Crítico**: Loss final = 3.99e-07 < 1e-06 (convergencia exitosa)

**🔥 Conclusión Crucial**: Para problemas con Neural Networks + measured data, el preconditioning **NO es opcional sino ESENCIAL**. Sin él, el solver híbrido falla completamente.

**Uso**: Demostrar la importancia crítica del preconditioning en problemas PINN con datos medidos

#### Example 7: PINN Híbrido con TODAS las Neural Networks (Caso más complejo)

**Base Example (sin preconditioning):**
```bash
cd FEM/python/examples/json && python generic.py example7.json
```
- **Solver**: Híbrido (GD → GD finalization para NNs)
- **Material**: **E,A,ρ = NNs independientes** (3 redes neurales)
  - Young: NN(x,y,λ) - 2×20 neuronas
  - Area: NN(x,y,λ) - 2×15 neuronas  
  - Density: NN(x,y,λ) - 2×10 neuronas
- **Measured Data**: Desplazamientos objetivo en nodos [1,2,3]
- **Preconditioning**: ❌ Deshabilitado (`"preconditioning": false`)
- **Performance**:
  - **✅ ÉXITO**: Converge los 10 incrementos (caso más complejo resuelto)
  - Incremento 1: ~1,900 iteraciones (3 NNs learning simultaneously)
  - Incrementos 2-10: ~79 iteraciones (warm start eficiente)
  - **Tiempo**: ~24.2 segundos

**Variante con Preconditioning:**
```bash 
cd FEM/python/examples/json && python generic.py example7-P.json
```
- **Solver**: Híbrido completo con preconditioning habilitado
- **Material**: Igual (E,A,ρ = 3 NNs independientes)
- **Preconditioning**: ✅ Habilitado (`"preconditioning": true`)
- **Estrategia**: Fase 1 (GD precon ~300 iter) → Fase 2 (GD main ~936 iter) 
- **Performance**:
  - **✅ ÉXITO MEJORADO**: Converge los 10 incrementos con mayor eficiencia
  - Incremento 1: 1,236 iteraciones totales (300 precon + 936 main)
  - Incrementos 2-10: ~129 iteraciones c/u (warm start + preconditioning)
  - **🚀 Tiempo**: ~10.5 segundos (**56% más rápido que sin preconditioning**)

**💡 Conclusión para Casos Complejos**: En problemas con múltiples Neural Networks (3+ NNs), el preconditioning proporciona mejoras dramáticas de performance (>50% reducción de tiempo), demostrando su valor en los casos más desafiantes de PINN.

**Uso**: Caso límite que demuestra el máximo beneficio del preconditioning en problemas híbridos multi-NN

#### Example 8: Full Newton-Raphson sin Neural Networks

```bash
cd FEM/python/examples/json && python generic.py example8.json
```
- **Solver**: Full Newton-Raphson con Hessiano (`"method": "full-nr"`)
- **Material**: Propiedades escalares constantes (E=A=ρ=1.0)
- **Objetivo**: Verificar que `full-nr` sin NNs produce resultados idénticos a `nr` clásico
- **Comportamiento**: Cuando `has_nn = False`, full-nr **delega automáticamente** a `solve_nr()` 
- **Performance**: 
  - 1 iteración por incremento (problema lineal)
  - **Resultado idéntico** a Example 1 (mismo desplazamiento, mismo tiempo)
  - **Tiempo**: ~0.6 segundos

**💡 Verificación**: Demuestra que Full Newton-Raphson es equivalente a Newton-Raphson clásico cuando no hay parámetros a optimizar. Para problemas sin NNs, ambos métodos resuelven el mismo sistema: `K·u = F`

**Uso**: Validación de la implementación de full-nr y comparación de solvers

#### Example 9: Full Newton-Raphson con Neural Network

```bash
cd FEM/python/examples/json && python generic.py example9.json
```
- **Solver**: Full Newton-Raphson con Hessiano completo (`"method": "full-nr"`)
- **Material**: E = NN(x,y,λ) con A,ρ escalares
  - Young: NN(x,y,λ) - 2×10 neuronas, input_dim=3
- **Measured Data**: Desplazamientos objetivo en nodos [1,2,3]
- **Método**: Calcula Hessiano completo [H_uu, H_uθ, H_θu, H_θθ] para convergencia cuadrática
- **Performance**:
  - **NN parameters**: 161 parámetros totales (6 tensores)
  - **Intenta calcular Hessiano**: `3×3 (DOFs) + 161×161 (NN params)`
  - **Fallback a GD**: Hessiano complejo → usa gradient descent
  - **Iteraciones**: ~1000 (variable según convergencia)
  - **Tiempo**: Variable (computacionalmente costoso)

**⚠️ Nota sobre Full Newton-Raphson**: El cálculo del Hessiano completo es extremadamente costoso computacionalmente. Para la mayoría de problemas PINN, el **solver híbrido** (Example 6-P, 7-P) ofrece mejor balance entre convergencia y costo computacional.

**💡 Full NR vs Híbrido**:
- **Full NR**: Convergencia cuadrática teórica, pero Hessiano muy costoso (O(n²) memoria y tiempo)
- **Híbrido**: Aproximación eficiente que combina GD (económico) + NR parcial (preciso)
- **Recomendación**: Usar híbrido con preconditioning para problemas PINN reales

**Uso**: Demostración académica de Full Newton-Raphson con NNs; no recomendado para producción

### Sistema de Warm Start

El solver implementa un sistema de **inicialización inteligente** que mejora dramáticamente la performance:

```
❄️  Incremento 1: Cold start (u = zeros)
🔥 Incrementos 2-10: Warm start (u = solución_anterior)
```

### Comparación de Solvers con Neural Networks

Esta sección compara diferentes estrategias de optimización para problemas PINN donde el módulo de Young es una Neural Network: **E = NN(x,y,λ)**

**Todos los ejemplos comparten la misma configuración de NN:**
- Young: NN(x,y,λ) - 2 capas × 20 neuronas (161 parámetros)
- Area: Escalar (1.0)
- Density: Escalar (1.0)
- Measured data: Desplazamientos objetivo en nodos [1,2,3]

#### Tabla Comparativa: Solvers con Young=NN

| **Ejemplo** | **Solver** | **Preconditioning** | **Iteraciones** | **Tiempo** | **Status** | **Eficiencia** |
|-------------|------------|---------------------|-----------------|------------|------------|----------------|
| **Example 3** | Gradient Descent | ❌ No | ~2,200 | 13.0s | ✅ SUCCESS | Baseline |
| **Example 3-P** | Gradient Descent | ✅ Sí | ~2,638 | 9.0s | ✅ SUCCESS | **31% más rápido** |
| **Example 6** | Híbrido (GD→GD) | ❌ No | 2,000 | 7.6s | ❌ **FAILED** | No converge |
| **Example 6-P** | Híbrido (GD→GD) | ✅ Sí | ~900 | **7.0s** | ✅ SUCCESS | **🚀 46% más rápido** |
| **Example 9** | Full Newton-Raphson | N/A (Hessiano) | ~1,000 | **60s** | ⚠️ Costoso | **❌ 361% más lento** |

#### Resumen Rápido

**🏆 Ganador: Example 6-P (Híbrido con Preconditioning)**
- ⚡ Más rápido: 7.0 segundos
- 🎯 Menos iteraciones: ~900
- ✅ Convergencia garantizada (vs Example 6 que falla)
- 📊 46% más rápido que GD puro (Example 3)

**🥈 Segunda opción: Example 3-P (GD con Preconditioning)**
- ⏱️ 9.0 segundos (29% más lento que Example 6-P)
- ✅ Confiable y estable
- 🔄 31% mejora sobre GD puro

**❌ Evitar:**
- **Example 6**: Falla completamente sin preconditioning
- **Example 9**: 8.6x más lento que Example 6-P (prohibitivo)

**💡 Conclusión Clave:** El preconditioning es la diferencia entre **éxito y fracaso** para métodos híbridos con NN.

#### Comparación Visual de Performance

**Tiempo de ejecución (menor es mejor):**
```
Example 6-P:  ████████ 7.0s  🏆 ÓPTIMO
Example 3-P:  ██████████ 9.0s  (+29%)
Example 6:    ████████▓▓ 7.6s  ❌ FALLA (no converge)
Example 3:    ██████████████ 13.0s  (+86%)
Example 9:    ████████████████████████████████████████████████████████████ 60s  (+757%)
```

**Iteraciones totales (menor es mejor):**
```
Example 6-P:  ██████████ 900 iter  🏆 MÁS EFICIENTE
Example 9:    ███████████ 1,000 iter  (pero 8x más lento)
Example 6:    ██████████████████████ 2,000 iter  ❌ FALLA
Example 3:    ████████████████████████ 2,200 iter  ✅ Converge
Example 3-P:  ████████████████████████████ 2,638 iter  ✅ Converge (31% más rápido que 3)
```

**Veredicto:**
- ⚡ **Velocidad**: 6-P (7s) > 3-P (9s) > 3 (13s) >>> 9 (60s)
- 🎯 **Iteraciones**: 6-P (900) > 9 (1000) > 3 (2200) > 3-P (2638)
- ✅ **Robustez**: 6-P = 3-P = 3 > 9 >> 6 (falla)
- 🏆 **Balance óptimo**: **Example 6-P** - mejor velocidad + garantía de convergencia

#### Análisis Detallado por Solver

**1. Example 3: Gradient Descent Puro**
- **Características**: 
  - Optimización de primer orden únicamente
  - Sin estrategia híbrida, solo GD end-to-end
  - Convergencia gradual pero robusta
- **Ventajas**: 
  - ✅ Simple y estable
  - ✅ No requiere configuración especial
  - ✅ Garantiza convergencia (aunque lenta)
- **Desventajas**: 
  - ❌ **~2,200 iteraciones** (más lento)
  - ❌ Tiempo: ~13 segundos (2x más que híbrido con precon)
- **Cuándo usar**: Baseline o cuando otros métodos fallen

**2. Example 3-P: Gradient Descent con Preconditioning**
- **Características**:
  - Fase 1: GD preconditioning (tolerancia 1e-4)
  - Fase 2: GD finalization (tolerancia 1e-6)
  - Warm-up inicial antes de refinamiento
- **Performance**:
  - ✅ **~2,638 iteraciones** totales
  - ✅ **~9 segundos** (**31% más rápido** que GD sin precon)
  - Incremento 1: ~1,707 iter (cold start con NN learning)
  - Incrementos 2-10: ~84-196 iter (warm start)
- **Ventajas**:
  - ✅ Mejora notable sobre GD puro
  - ✅ Convergencia garantizada
  - ✅ Buena inicialización para NN
- **Desventajas**:
  - Más iteraciones totales que híbrido (pero más rápido que GD sin precon)
- **Cuándo usar**: Alternativa confiable cuando híbrido no está disponible

**3. Example 6: Híbrido sin Preconditioning**
- **Características**:
  - Intenta combinar GD inicial + finalization
  - Sin warm-up (preconditioning deshabilitado)
  - Measured data + NN = problema difícil
- **Resultado**:
  - ❌ **FALLO TOTAL**: No converge en 2,000 iteraciones
  - Loss final: 6.578e-06 (no alcanza tolerancia 1e-06)
  - Solo completa 1 de 10 incrementos
- **Conclusión**: **Preconditioning es CRÍTICO** para problemas híbridos con NN + measured data

**4. Example 6-P: Híbrido con Preconditioning ⭐ RECOMENDADO**
- **Características**:
  - Fase 1: GD preconditioning (~300 iter, tolerancia 1e-4)
  - Fase 2: GD finalization (~581 iter, tolerancia 1e-6)
  - Warm-up permite mejor inicialización
- **Ventajas**:
  - ✅ **Converge exitosamente** (vs fallo sin precon)
  - ✅ **~900 iteraciones** (58% menos que GD puro, 66% menos que GD-P)
  - ✅ **~7.0 segundos** (46% más rápido que GD puro, 22% más rápido que GD-P)
  - ✅ Balance óptimo: velocidad + robustez
- **Desventajas**: 
  - Requiere configuración de preconditioning
- **Cuándo usar**: **Problemas PINN reales con NN + measured data**

**5. Example 9: Full Newton-Raphson**
- **Características**:
  - Calcula Hessiano completo [H_uu, H_uθ, H_θu, H_θθ]
  - 161 parámetros NN → Hessiano 161×161
  - Convergencia cuadrática teórica
- **Realidad**:
  - ⚠️ Hessiano **extremadamente costoso** computacionalmente
  - Fallback a gradient descent por complejidad
  - O(n²) memoria y tiempo para segundo orden
  - **~60 segundos** (10x más lento que métodos prácticos)
- **Ventajas**:
  - ✅ Convergencia cuadrática en teoría
  - ✅ Demostración académica completa
- **Desventajas**:
  - ❌ **Costo computacional prohibitivo** (10x más lento)
  - ❌ No práctico para problemas reales
  - ❌ Híbrido es más eficiente en práctica
- **Cuándo usar**: Investigación académica, NO producción

#### 🏆 Recomendaciones para Problemas PINN con Neural Networks

**Para Producción y Aplicaciones Reales:**
```python
# Configuración óptima (Example 6-P)
{
  "solver_type": "fem",
  "solver_config": {
    "method": "hybrid"  # GD → GD finalization
  },
  "pinn_config": {
    "preconditioning": true,  # ✅ CRÍTICO para NN + data
    "tolerance": 1e-6,
    "max_iterations": 1000
  }
}
```

**Orden de Preferencia:**
1. **🥇 Híbrido + Preconditioning** (Example 6-P): Mejor balance velocidad/robustez - **7.0s**
2. **🥈 Gradient Descent + Preconditioning** (Example 3-P): Confiable - **9.0s** (31% mejora vs GD puro)
3. **🥉 Gradient Descent puro** (Example 3): Fallback sólido - **13.0s** (baseline)
4. **❌ Híbrido sin Preconditioning** (Example 6): **NO usar** (falla totalmente, no converge)
5. **❌ Full Newton-Raphson** (Example 9): Solo investigación académica - **60s** (8.6x más lento)

**Tabla de Decisión Rápida:**

| Si necesitas... | Usa... | Tiempo | Razón |
|----------------|--------|--------|-------|
| **Máxima velocidad + robustez** | Example 6-P | 7.0s | 🏆 Balance óptimo |
| **Alternativa confiable** | Example 3-P | 9.0s | Estable, 31% mejor que GD puro |
| **Máxima simplicidad** | Example 3 | 13.0s | Simple, siempre converge |
| **Investigación académica** | Example 9 | 60s | Hessiano completo (demo teórica) |
| **❌ Nunca usar** | Example 6 | - | Falla sin preconditioning |

**Datos de Performance Comparativa:**
- **Velocidad relativa** (vs Example 3 baseline):
  - Example 6-P: **46% más rápido** ⚡
  - Example 3-P: **31% más rápido** ✅
  - Example 3: 0% (baseline)
  - Example 9: **361% más lento** ❌
  
- **Convergencia**:
  - ✅ Example 3, 3-P, 6-P: Convergen exitosamente
  - ❌ Example 6: Falla (solo 1/10 incrementos)
  - ⚠️ Example 9: Converge pero prohibitivamente costoso

**Mejoras de Performance del Preconditioning:**
- **Con 1 NN (Young)**: Diferencia entre éxito y fallo completo
- **Con 3 NNs (E,A,ρ)**: 56% reducción de tiempo (Example 7-P)
- **Conclusión**: Preconditioning es **esencial** para problemas multi-NN

---

### Comparación de Solvers con 3 Neural Networks (Multi-Property)

Esta sección compara estrategias para problemas PINN con **todas las propiedades** como NNs: **E = NN(x,y,λ), A = NN(x,y,λ), ρ = NN(x,y,λ)**

**Configuración común de NNs:**
- Young: NN(x,y,λ) - 2 capas × 20 neuronas = 521 parámetros
- Area: NN(x,y,λ) - 2 capas × 15 neuronas = 226 parámetros
- Density: NN(x,y,λ) - 2 capas × 10 neuronas = 91 parámetros
- **Total: 838 parámetros NN** (vs 161 con 1 NN)
- Measured data: Desplazamientos en nodos [1,2,3]

#### Tabla Comparativa: Solvers con 3 NNs

| **Ejemplo** | **Solver** | **Preconditioning** | **Iteraciones** | **Tiempo** | **Status** | **Eficiencia** |
|-------------|------------|---------------------|-----------------|------------|------------|----------------|
| **Example 4** | Gradient Descent | ❌ No | ~3,500 | **3min** (~180s) | ✅ SUCCESS | Baseline |
| **Example 4-P** | Gradient Descent | ✅ Sí | ~2,126 | **18s** | ✅ SUCCESS | **🚀 90% más rápido** |
| **Example 7** | Híbrido (GD→GD) | ❌ No | ~79 | 24.2s | ✅ SUCCESS | 87% más rápido |
| **Example 7-P** | Híbrido (GD→GD) | ✅ Sí | ~1,236 | **10.5s** | ✅ SUCCESS | **🏆 94% más rápido** |
| **Example 10** | Full Newton-Raphson | N/A (Hessiano 838×838) | ~1,000 | **>120s** | ⚠️ Prohibitivo | Hessiano inviable |

#### Resumen Rápido: Multi-Property PINN

**🏆 Ganador: Example 7-P (Híbrido + Preconditioning)**
- ⚡ Más rápido: **10.5 segundos** (94% mejor que GD puro)
- 🎯 Balance óptimo para 3 NNs (838 parámetros)
- ✅ Convergencia garantizada en todos los incrementos
- 📊 56% más rápido que híbrido sin preconditioning

**🥈 Segunda opción: Example 4-P (GD + Preconditioning)**
- ⏱️ **18 segundos** (90% mejor que GD puro)
- ✅ Muy confiable, simplicidad máxima
- 🔄 10x más rápido que GD sin preconditioning

**❌ Evitar:**
- **Example 4 sin precon**: 3 minutos (10x más lento)
- **Example 10 (Full NR)**: >120s con Hessiano 838×838 (computacionalmente inviable)

#### Comparación Visual: Multi-Property

**Tiempo de ejecución (menor es mejor):**
```
Example 7-P:  ███████████ 10.5s  🏆 ÓPTIMO (94% mejora)
Example 4-P:  ████████████████████ 18s  (90% mejora)
Example 7:    ██████████████████████████ 24.2s  (87% mejora)
Example 10:   ████████████████████████████████████████████████████████████████████ >120s  ❌
Example 4:    ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 180s (3min)
```

**Iteraciones totales:**
```
Example 7:    ████ 79 iter    (estrategia híbrida muy eficiente)
Example 7-P:  ████████████████████████████████ 1,236 iter  (más iter pero 2x más rápido)
Example 4-P:  ████████████████████████████████████████████ 2,126 iter  (10x más rápido que sin precon)
Example 4:    ████████████████████████████████████████████████████████████████████ 3,500 iter
Example 10:   ████████████████████ ~1,000 iter  (pero cada una extremadamente costosa)
```

**Veredicto Multi-Property:**
- ⚡ **Velocidad**: 7-P (10.5s) >> 4-P (18s) >> 7 (24.2s) >>>>> 4 (180s)
- 🎯 **Eficiencia**: 7-P (1,236 iter en 10.5s) vs 4-P (2,126 iter en 18s)
- ✅ **Robustez**: Todos convergen (excepto 10 que es inviable)
- 🏆 **Recomendación**: **Example 7-P** - óptimo absoluto para 3 NNs

#### Análisis Detallado por Solver (3 NNs)

**1. Example 4: Gradient Descent Puro (3 NNs)**
- **Características**:
  - 838 parámetros NN totales
  - Incremento 1: ~2,755 iteraciones (cold start, 3 NNs learning)
  - Incrementos 2-10: ~80-150 iteraciones cada uno
- **Performance**: ~3,500 iteraciones, **~180 segundos (3 minutos)**
- **Conclusión**: Extremadamente lento sin preconditioning
- **Cuándo usar**: Nunca - siempre preferir 4-P

**2. Example 4-P: Gradient Descent + Preconditioning (3 NNs) ✅**
- **Características**:
  - Fase preconditioning: ~1,187 iter en incremento 1
  - Fase finalization: mejor convergencia
  - Warm start muy efectivo en incrementos 2-10
- **Performance**: ~2,126 iteraciones, **~18 segundos**
- **Mejora**: **90% reducción de tiempo** vs Example 4
- **Ventajas**:
  - ✅ 10x más rápido que sin preconditioning
  - ✅ Máxima simplicidad (solo GD)
  - ✅ Muy confiable
- **Cuándo usar**: Alternativa robusta cuando híbrido no está disponible

**3. Example 7: Híbrido sin Preconditioning (3 NNs)**
- **Características**:
  - Sorprendentemente, **converge** (a diferencia de Example 6 con 1 NN)
  - Solo ~79 iteraciones reportadas (última incremental)
  - 3 NNs = problema más complejo pero estrategia híbrida funciona
- **Performance**: ~79 iteraciones (último incremento), **~24.2 segundos**
- **Observación**: A pesar de pocas iteraciones, tiempo mayor que esperado
- **Cuándo usar**: Caso académico - siempre preferir 7-P

**4. Example 7-P: Híbrido + Preconditioning (3 NNs) 🏆 RECOMENDADO**
- **Características**:
  - Estrategia híbrida completa con preconditioning
  - ~1,236 iteraciones totales (más que 7, pero mucho más rápido)
  - Warm-up crucial para 838 parámetros NN
- **Performance**: ~1,236 iteraciones, **~10.5 segundos**
- **Mejora**: **56% reducción de tiempo** vs Example 7
- **Ventajas**:
  - ✅ **Más rápido** de todos los métodos prácticos
  - ✅ **94% más rápido** que GD puro (Example 4)
  - ✅ 42% más rápido que GD con preconditioning (Example 4-P)
  - ✅ Balance perfecto: velocidad + robustez
- **Cuándo usar**: **Siempre** para problemas PINN con múltiples NNs

**5. Example 10: Full Newton-Raphson (3 NNs) ❌ NO USAR**
- **Características**:
  - Intenta calcular Hessiano completo: 3×3 (u) + 838×838 (θ)
  - Matriz Hessiana de **~838×838 = 702,244 elementos**
  - Cada iteración requiere computar segundas derivadas para 838 parámetros
- **Realidad**:
  - ⚠️ **Computacionalmente prohibitivo**
  - Fallback a gradient descent (igual que Example 9)
  - Tiempo estimado: **>120 segundos** (10x más lento que 7-P)
  - Memoria: O(838²) ≈ 5.3MB solo para Hessiano NN
- **Conclusión**: **Totalmente inviable para producción**
- **Cuándo usar**: Solo demostración académica de limitaciones

#### 🏆 Recomendaciones para Problemas Multi-NN (3 NNs)

**Para Producción:**
```json
// Configuración óptima: Example 7-P
{
  "solver_type": "pinn-hybrid",
  "pinn_config": {
    "preconditioning": true,  // ✅ CRÍTICO para 3 NNs
    "tolerance": 1e-6,
    "max_iterations": 2000
  },
  "nn_config": {
    "young": {"enabled": true, "neurons_per_layer": 20},
    "area": {"enabled": true, "neurons_per_layer": 15},
    "density": {"enabled": true, "neurons_per_layer": 10}
  }
}
```

**Orden de Preferencia (3 NNs):**
1. **🥇 Híbrido + Preconditioning** (Example 7-P): **10.5s** - Balance óptimo
2. **🥈 GD + Preconditioning** (Example 4-P): **18s** - Alternativa confiable
3. **🥉 Híbrido sin Preconditioning** (Example 7): **24.2s** - Funciona pero subóptimo
4. **❌ GD sin Preconditioning** (Example 4): **180s** - 10x más lento (evitar)
5. **❌ Full Newton-Raphson** (Example 10): **>120s** - Inviable (Hessiano 838×838)

**Comparación de Velocidades:**
| Solver | Tiempo | Relativo a 7-P |
|--------|---------|----------------|
| **Example 7-P** | 10.5s | 1.0x 🏆 |
| **Example 4-P** | 18s | 1.7x |
| **Example 7** | 24.2s | 2.3x |
| **Example 10** | >120s | >11x ❌ |
| **Example 4** | 180s | 17x ❌ |

**Impacto del Preconditioning en Multi-NN:**
- **Con GD**: 90% reducción de tiempo (4-P vs 4)
- **Con Híbrido**: 56% reducción de tiempo (7-P vs 7)
- **Conclusión**: Preconditioning es **absolutamente esencial** para problemas con múltiples NNs

**Comparación 1 NN vs 3 NNs:**
| Configuración | 1 NN (161 params) | 3 NNs (838 params) | Factor | Escalabilidad |
|---------------|-------------------|---------------------|--------|---------------|
| **GD** | 13s (Ex 3) | 180s (Ex 4) | 13.8x más lento | ❌ Pésima |
| **GD+Precon** | 9s (Ex 3-P) | 18s (Ex 4-P) | 2.0x más lento | ✅ Buena |
| **Híbrido+Precon** | 7s (Ex 6-P) | 10.5s (Ex 7-P) | 1.5x más lento | 🏆 Excelente |
| **Full NR** | 60s (Ex 9) | >120s (Ex 10) | >2.0x más lento | ⚠️ Prohibitivo |

**Análisis de Escalabilidad:**
- **Mejor escalado**: Híbrido+Precon (1.5x) - prácticamente lineal con el número de NNs 🏆
- **Buen escalado**: GD+Precon (2.0x) - escalado razonable
- **Mal escalado**: GD puro (13.8x) - colapsa con múltiples NNs ❌
- **Inviable**: Full NR (>2.0x pero desde 60s base) - ambos casos prohibitivamente costosos ❌

**Conclusión clave**: El solver híbrido con preconditioning **escala mucho mejor** con el número de NNs que gradient descent puro. Full Newton-Raphson es inviable incluso con 1 NN (60s) y empeora dramáticamente con 3 NNs (>120s).

---

### Sistema de Preconditioning

Los solvers GD e Híbrido incluyen un sistema de **preconditioning opcional** que acelera la convergencia:

**Estrategia de Two-Phase**:
```
🏃 Fase 1: Preconditioning (tolerancia relajada, warm-up)
🎯 Fase 2: Main solve (tolerancia estricta, convergencia final)
```

**Configuración**:
```json
{
  "pinn_config": {
    "preconditioning": true,  // ✅ Habilitar preconditioning
    "tolerance": 1e-6         // Tolerancia final
  }
}
```

**Cuándo usar preconditioning**:
- ✅ **Recomendado**: Problemas GD puros (Example 2-P vs 2: 45% más rápido)
- ❌ **Innecesario**: Problemas lineales sin NNs (Example 5 vs 5-P: directo es 4x más rápido)
- 🚨 **CRÍTICO**: Problemas con NNs + measured data (Example 6-P vs 6: éxito vs fallo total)
- 🚀 **EXCELENTE**: Casos multi-NN complejos (Example 7-P vs 7: 56% más rápido)
- ✅ **Útil**: Problemas no-lineales con NNs (robustez + velocidad)

**Beneficios por tipo de solver**:
- **Newton-Raphson**: No aplica (convergencia ya óptima)
- **Gradient Descent**: Mejora significativa (30-50% reducción de tiempo)
- **Híbrido con 1 NN**: 🚨 **ESENCIAL** - sin preconditioning = fallo completo
- **Híbrido con 3+ NNs**: 🚀 **EXCELENTE** - 56% reducción de tiempo en casos complejos
- **Híbrido sin NNs**: Overhead innecesario (usar directo)

### Ejecutar Todos los Ejemplos

```bash
# Ejecutar suite completa de ejemplos base
cd FEM/python/examples/json
python generic.py example1.json
python generic.py example2.json  
python generic.py example3.json
python generic.py example4.json
python generic.py example5.json
python generic.py example6.json
python generic.py example7.json

# Probar variantes con preconditioning
python generic.py example2-P.json  # GD con preconditioning
python generic.py example5-P.json  # Híbrido con preconditioning
python generic.py example6-P.json  # Híbrido+NN con preconditioning (CRÍTICO)
python generic.py example7-P.json  # Híbrido+3NNs con preconditioning (MÁXIMO BENEFICIO)

# Comparar resultados y performance
ls -la *.res.json
```

**Benchmarks Recomendados**:
```bash
# Comparar GD con/sin preconditioning
time python generic.py example2.json   # ~5.6s
time python generic.py example2-P.json # ~3.1s (45% más rápido)

# Comparar híbrido con/sin preconditioning  
time python generic.py example5.json   # ~0.67s (óptimo directo)
time python generic.py example5-P.json # ~2.4s (overhead innecesario)

# CRÍTICO: Comparar híbrido+NN con/sin preconditioning
time python generic.py example6.json   # ~7.6s (❌ FALLA - no converge)
time python generic.py example6-P.json # ~7.0s (✅ ÉXITO - converge)

# MÁXIMO BENEFICIO: Casos complejos multi-NN
time python generic.py example7.json   # ~24.2s (✅ éxito lento)
time python generic.py example7-P.json # ~10.5s (🚀 56% MÁS RÁPIDO)
```

## API Endpoints

### POST /api/fem/solve

Resuelve problema FEM clásico.

**Request Body:**
```json
{
  "nodes": [
    {"x": 0.0, "y": 0.0, "fixed": true},
    {"x": 1.0, "y": 0.0, "fixed_y": true}
  ],
  "elements": [
    {"nodes": [0, 1]}
  ],
  "material": {
    "young": 210e9,
    "area": 0.01,
    "density": 7850
  },
  "loads": [0, 0, 1000, 0],
  "solver_config": {
    "tolerance": 1e-6,
    "max_iterations": 50,
    "n_increments": 10
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "displacements": [0, 0, 0.00047619, 0],
    "stresses": [100000000],
    "strains": [0.00047619],
    "converged": true,
    "convergence_history": [...]
  }
}
```

### POST /api/fem/solve-pinn

Resuelve problema inverso con PINN.

### GET /api/fem/info

Información sobre solvers disponibles.

## Troubleshooting

### El frontend no se conecta al backend

- Verifica que el backend esté corriendo en puerto 5000
- Verifica el proxy en `vite.config.js`

### Python error: "Module not found"

- Asegúrate de estar en el virtual environment
- Instala dependencies: `pip install numpy torch matplotlib`

### "Singular matrix" error

- Verifica condiciones de frontera (necesitas al menos 3 DOFs fijados para 2D)
- Revisa la geometría
- Reduce la carga aplicada

## Licencia

MIT
