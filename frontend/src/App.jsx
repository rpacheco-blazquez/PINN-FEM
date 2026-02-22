import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import FEMCanvas from './components/FEMCanvas'
import DataTable from './components/DataTable'
import PropertiesPanel from './components/PropertiesPanel'
import CalculateButton from './components/CalculateButton'

function App() {
  // Cargar desde localStorage si existe
  const loadFromStorage = (key, defaultValue) => {
    try {
      const saved = localStorage.getItem(key)
      return saved ? JSON.parse(saved) : defaultValue
    } catch (error) {
      console.error(`Error loading ${key} from localStorage:`, error)
      return defaultValue
    }
  }

  const defaultNnConfig = {
    young: { enabled: false, hiddenLayers: 2, neuronsPerLayer: 20 },
    area: { enabled: false, hiddenLayers: 2, neuronsPerLayer: 20 },
    density: { enabled: false, hiddenLayers: 2, neuronsPerLayer: 20 }
  }

  const [nodes, setNodes] = useState(() => loadFromStorage('fem_nodes', []))
  const [elements, setElements] = useState(() => loadFromStorage('fem_elements', []))
  const [nnConfig, setNnConfig] = useState(() => loadFromStorage('fem_nn_config', defaultNnConfig))
  const [displacements, setDisplacements] = useState(null)
  const [selectedItem, setSelectedItem] = useState(null)
  const [isCalculating, setIsCalculating] = useState(false)

  // Guardar automáticamente en localStorage cuando cambien nodes o elements
  useEffect(() => {
    localStorage.setItem('fem_nodes', JSON.stringify(nodes))
  }, [nodes])

  useEffect(() => {
    localStorage.setItem('fem_elements', JSON.stringify(elements))
  }, [elements])

  useEffect(() => {
    localStorage.setItem('fem_nn_config', JSON.stringify(nnConfig))
  }, [nnConfig])

  const handleCalculate = async (solverConfig) => {
    setIsCalculating(true)
    try {
      console.log('Calculating FEM solution...')
      console.log('Nodes:', nodes)
      console.log('Elements:', elements)
      console.log('Solver config:', solverConfig)

      // Construir vector de cargas (fuerzas aplicadas en nodos libres)
      const loads = []
      nodes.forEach(node => {
        const bcType = node.bcType || 'free'
        const bcValue = node.bcValue !== undefined ? node.bcValue : 0

        if (bcType === 'free') {
          // Nodo libre: aplicar fuerza
          loads.push(bcValue, 0)  // fx, fy (solo fx por ahora)
        } else {
          // Nodo fijo: fuerza cero (se manejará con BCs)
          loads.push(0, 0)
        }
      })

      // Obtener propiedades materiales del primer elemento (por ahora)
      // En el futuro, cada elemento puede tener propiedades diferentes
      const firstElement = elements[0] || {}
      const material = {
        young: firstElement.young !== undefined ? firstElement.young : 210e9,
        area: firstElement.area !== undefined ? firstElement.area : 0.01,
        density: firstElement.density !== undefined ? firstElement.density : 7850
      }

      // Preparar datos para el backend
      const problemData = {
        nodes: nodes.map(n => ({
          x: n.x,
          y: n.y,
          fixed: n.bcType === 'fixed' || n.fixed,
          fixed_x: false,
          fixed_y: false,
          measured_ux: n.measuredUx || 0,
          measured_uy: n.measuredUy || 0
        })),
        elements: elements.map(e => ({
          nodes: e.nodes
        })),
        material: material,
        loads: loads,
        solver_config: {
          tolerance: solverConfig.tolerance,
          max_iterations: solverConfig.maxIterations,
          n_increments: 10
        },
        nn_config: nnConfig,
        solver_type: solverConfig.solverType
      }

      // Usar el solver genérico que maneja FEM y PINN
      console.log('Sending to generic solver...')
      console.log('NN Config:', nnConfig)

      const response = await axios.post('/api/fem/solve-generic', problemData)

      // Procesar resultados
      if (response.data.success) {
        const result = response.data.result

        // Convertir displacements a formato del frontend
        const disp = []
        for (let i = 0; i < nodes.length; i++) {
          disp.push({
            node: i,
            ux: result.displacements[2 * i] || 0,
            uy: result.displacements[2 * i + 1] || 0
          })
        }

        setDisplacements(disp)

        // Mostrar resultados adicionales
        if (result.identified_params) {
          console.log('Parámetros identificados:', result.identified_params)
          alert(`Cálculo completado!\nYoung: ${result.identified_params.young.toExponential(3)} Pa\nÁrea: ${result.identified_params.area.toFixed(6)} m²`)
        } else {
          alert('Cálculo completado!')
        }
      } else {
        throw new Error(response.data.error || 'Error desconocido')
      }

    } catch (error) {
      console.error('Error calculating:', error)
      alert(`Error en el cálculo: ${error.response?.data?.error || error.message}`)
    } finally {
      setIsCalculating(false)
    }
  }

  const handleClearModel = () => {
    if (window.confirm('¿Estás seguro de que quieres limpiar todo el modelo? Esta acción no se puede deshacer.')) {
      setNodes([])
      setElements([])
      setDisplacements(null)
      setSelectedItem(null)
      setNnConfig(defaultNnConfig)
      localStorage.removeItem('fem_nodes')
      localStorage.removeItem('fem_elements')
      localStorage.removeItem('fem_nn_config')
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🔧 PINN-FEM CAD</h1>
        <p>Physics-Informed Neural Networks + Finite Element Method</p>
      </header>

      <div className="main-layout">
        {/* Canvas - Visualización de la malla y deformación */}
        <div className="canvas-section">
          <FEMCanvas
            nodes={nodes}
            setNodes={setNodes}
            elements={elements}
            setElements={setElements}
            displacements={displacements}
            selectedItem={selectedItem}
            setSelectedItem={setSelectedItem}
          />
        </div>

        {/* Tabla de datos - Coordenadas y conectividades */}
        <div className="table-section">
          <DataTable
            nodes={nodes}
            setNodes={setNodes}
            elements={elements}
            setElements={setElements}
            displacements={displacements}
            nnConfig={nnConfig}
            setNnConfig={setNnConfig}
          />
        </div>

        {/* Propiedades - Material, BCs, etc */}
        <div className="properties-section">
          <PropertiesPanel
            selectedItem={selectedItem}
            nodes={nodes}
            setNodes={setNodes}
            elements={elements}
            setElements={setElements}
          />
        </div>

        {/* Botón de cálculo */}
        <div className="calculate-section">
          <CalculateButton
            onCalculate={handleCalculate}
            isCalculating={isCalculating}
            disabled={nodes.length < 2 || elements.length < 1}
          />
          <button
            className="clear-model-btn"
            onClick={handleClearModel}
            disabled={isCalculating || (nodes.length === 0 && elements.length === 0)}
            title="Limpiar todo el modelo"
          >
            🗑️ Limpiar Modelo
          </button>
        </div>
      </div>
    </div>
  )
}

export default App
