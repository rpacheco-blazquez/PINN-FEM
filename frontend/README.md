# PINN-FEM CAD Frontend

Frontend React + Vite para el CAD de elementos finitos con Physics-Informed Neural Networks.

## 🚀 Características

### Layout en 4 Secciones:

```
┌─────────────────────┬──────────────┐
│                     │              │
│   Canvas (CAD)      │  Data Table  │
│   Visualización     │  Nodos/Elem  │
│                     │              │
├─────────────────────┼──────────────┤
│  Properties Panel   │  Calculate   │
│  Material, BCs      │  Button      │
└─────────────────────┴──────────────┘
```

### 1. **Canvas (Área de Dibujo)**
- ✅ Añadir nodos haciendo clic
- ✅ Conectar nodos para crear elementos
- ✅ Visualización de malla
- ✅ Visualización de deformaciones con shape functions
- ✅ Escala de deformación ajustable
- ✅ Grid de referencia
- ✅ Indicadores de soportes fijos

### 2. **Data Table (Tabla de Datos)**
- ✅ Edición estilo Excel
- ✅ Tabs: Nodos / Elementos / Resultados
- ✅ Editar coordenadas (X, Y)
- ✅ Marcar nodos como fijos
- ✅ Editar conectividades
- ✅ Ver desplazamientos calculados

### 3. **Properties Panel (Propiedades)**
- ✅ Muestra info del item seleccionado
- ✅ Coordenadas de nodos
- ✅ Longitud de elementos
- 🔜 Editar propiedades de material
- 🔜 Aplicar cargas
- 🔜 Condiciones de contorno

### 4. **Calculate Button (Cálculo)**
- ✅ Botón de cálculo FEM
- ✅ Validación de modelo
- ✅ Opciones de solver:
  - FEM Clásico
  - PINN (Gradient Descent)
  - PINN (Newton-Raphson)
- 🔜 Conexión con backend Python

## 📦 Instalación

```bash
cd frontend
npm install
npm run dev
```

Abre [http://localhost:3000](http://localhost:3000)

## 🎮 Cómo Usar

1. **Crear Nodos:**
   - Selecciona modo "⬤ Nodo"
   - Haz clic en el canvas para añadir nodos
   - Marca nodos como fijos en la tabla

2. **Crear Elementos:**
   - Selecciona modo "─ Elemento"
   - Haz clic en dos nodos para conectarlos

3. **Editar en Tabla:**
   - Cambia coordenadas directamente
   - Añade/elimina nodos y elementos

4. **Calcular:**
   - Presiona "🚀 Calcular FEM"
   - Ve los resultados en la tabla y canvas

## 🔧 Tecnologías

- **React 18** - UI Framework
- **Vite** - Build tool
- **HTML Canvas** - Renderizado 2D
- **Axios** - HTTP client (para backend)

## 🎨 Características del Canvas

- **Shape Functions:** Interpolación lineal entre nodos
- **Deformación:** Escalado ajustable para visualizar mejor
- **Colores:**
  - 🔵 Nodos normales
  - 🟡 Nodos seleccionados
  - 🟢 Configuración deformada
  - ⚪️ Configuración original (línea punteada)
  - 🔴 Soportes fijos

## 🔜 Próximos Pasos

- [ ] Conexión con backend Python FEM
- [ ] API REST para cálculo
- [ ] Edición de propiedades materiales
- [ ] Aplicación de cargas (fuerzas, momentos)
- [ ] Exportar/importar modelos (JSON)
- [ ] Visualización de tensiones
- [ ] Elementos 3D (Three.js)
- [ ] PINN: identificación de parámetros

## 📝 Estructura del Código

```
frontend/
├── src/
│   ├── components/
│   │   ├── FEMCanvas.jsx          # Canvas de dibujo
│   │   ├── DataTable.jsx          # Tabla de datos
│   │   ├── PropertiesPanel.jsx   # Panel de propiedades
│   │   └── CalculateButton.jsx   # Botón de cálculo
│   ├── App.jsx                    # App principal
│   └── main.jsx                   # Entry point
├── package.json
└── vite.config.js
```

## 🤝 Integración Backend

El frontend está preparado para comunicarse con el backend Python:

```javascript
// TODO: Implementar en handleCalculate()
const response = await axios.post('/api/solve', {
  nodes: nodes,
  elements: elements,
  material: { young: 2.1e11, area: 1e-4 },
  solver: 'fem' // o 'pinn-gd', 'pinn-nr'
})
```
