import { useState } from 'react'
import './DataTable.css'

const DataTable = ({ nodes, setNodes, elements, setElements, displacements, nnConfig, setNnConfig }) => {
  const [activeTab, setActiveTab] = useState('nodes') // 'nodes', 'elements', 'results'

  const handleNodeChange = (index, field, value) => {
    const newNodes = [...nodes]
    if (field === 'x' || field === 'y') {
      newNodes[index][field] = parseFloat(value) || 0
    } else if (field === 'fixed') {
      newNodes[index][field] = value
      // Sincronizar con bcType para compatibilidad
      newNodes[index].bcType = value ? 'fixed' : 'free'
    } else if (field === 'measuredUx' || field === 'measuredUy') {
      newNodes[index][field] = parseFloat(value) || 0
    }
    setNodes(newNodes)
  }

  const handleNnConfigChange = (property, field, value) => {
    const newConfig = {
      ...nnConfig,
      [property]: {
        ...nnConfig[property],
        [field]: field === 'enabled' ? value : parseInt(value) || 0
      }
    }
    setNnConfig(newConfig)
  }

  const handleElementChange = (index, nodeIdx, value) => {
    const newElements = [...elements]
    newElements[index].nodes[nodeIdx] = parseInt(value) || 0
    setElements(newElements)
  }

  const addNode = () => {
    setNodes([...nodes, { x: 0, y: 0, fixed: false, bcType: 'free', bcValue: 0, measuredUx: 0, measuredUy: 0 }])
  }

  const deleteNode = (index) => {
    const newNodes = nodes.filter((_, i) => i !== index)
    // Actualizar elementos que referencian nodos eliminados
    const newElements = elements
      .filter(elem => !elem.nodes.includes(index))
      .map(elem => ({
        nodes: elem.nodes.map(n => n > index ? n - 1 : n)
      }))
    setNodes(newNodes)
    setElements(newElements)
  }

  const addElement = () => {
    if (nodes.length >= 2) {
      setElements([...elements, {
        nodes: [0, 1],
        young: 210e9,    // Pa (acero por defecto)
        area: 0.01,      // m² por defecto
        density: 7850    // kg/m³ (acero)
      }])
    }
  }

  const deleteElement = (index) => {
    setElements(elements.filter((_, i) => i !== index))
  }

  return (
    <div className="data-table-container">
      <div className="table-tabs">
        <button
          className={activeTab === 'nodes' ? 'active' : ''}
          onClick={() => setActiveTab('nodes')}
        >
          📍 Nodos ({nodes.length})
        </button>
        <button
          className={activeTab === 'elements' ? 'active' : ''}
          onClick={() => setActiveTab('elements')}
        >
          ━ Elementos ({elements.length})
        </button>
        <button
          className={activeTab === 'data' ? 'active' : ''}
          onClick={() => setActiveTab('data')}
        >
          📝 Data
        </button>
        {displacements && (
          <button
            className={activeTab === 'results' ? 'active' : ''}
            onClick={() => setActiveTab('results')}
          >
            📊 Resultados
          </button>
        )}
      </div>

      <div className="table-content">
        {activeTab === 'nodes' && (
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>X</th>
                  <th>Y</th>
                  <th>Fijo</th>
                  <th>Acción</th>
                </tr>
              </thead>
              <tbody>
                {nodes.map((node, idx) => (
                  <tr key={idx}>
                    <td>{idx}</td>
                    <td>
                      <input
                        type="number"
                        step="0.1"
                        value={node.x.toFixed(2)}
                        onChange={(e) => handleNodeChange(idx, 'x', e.target.value)}
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        step="0.1"
                        value={node.y.toFixed(2)}
                        onChange={(e) => handleNodeChange(idx, 'y', e.target.value)}
                      />
                    </td>
                    <td>
                      <input
                        type="checkbox"
                        checked={node.fixed || false}
                        onChange={(e) => handleNodeChange(idx, 'fixed', e.target.checked)}
                      />
                    </td>
                    <td>
                      <button
                        className="delete-btn"
                        onClick={() => deleteNode(idx)}
                        title="Eliminar nodo"
                      >
                        🗑️
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button className="add-row-btn" onClick={addNode}>
              ➕ Añadir Nodo
            </button>
          </div>
        )}

        {activeTab === 'elements' && (
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Nodo 1</th>
                  <th>Nodo 2</th>
                  <th>Acción</th>
                </tr>
              </thead>
              <tbody>
                {elements.map((elem, idx) => (
                  <tr key={idx}>
                    <td>{idx}</td>
                    <td>
                      <input
                        type="number"
                        min="0"
                        max={nodes.length - 1}
                        value={elem.nodes[0]}
                        onChange={(e) => handleElementChange(idx, 0, e.target.value)}
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        min="0"
                        max={nodes.length - 1}
                        value={elem.nodes[1]}
                        onChange={(e) => handleElementChange(idx, 1, e.target.value)}
                      />
                    </td>
                    <td>
                      <button
                        className="delete-btn"
                        onClick={() => deleteElement(idx)}
                        title="Eliminar elemento"
                      >
                        🗑️
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button
              className="add-row-btn"
              onClick={addElement}
              disabled={nodes.length < 2}
            >
              ➕ Añadir Elemento
            </button>
          </div>
        )}

        {activeTab === 'data' && (
          <div className="table-wrapper">
            {/* Sección de configuración NN */}
            <div className="nn-config-section">
              <h3>Propiedades Desconocidas (NN)</h3>
              <p className="help-text">Selecciona qué propiedades materiales aproximar con redes neuronales</p>

              <div className="nn-property-config">
                {Object.entries(nnConfig || {}).map(([property, config]) => (
                  <div key={property} className="nn-property-row">
                    <div className="nn-property-checkbox">
                      <input
                        type="checkbox"
                        id={`nn-${property}`}
                        checked={config.enabled}
                        onChange={(e) => handleNnConfigChange(property, 'enabled', e.target.checked)}
                      />
                      <label htmlFor={`nn-${property}`}>
                        {property === 'young' && 'Módulo de Young (E)'}
                        {property === 'area' && 'Área transversal (A)'}
                        {property === 'density' && 'Densidad (ρ)'}
                      </label>
                    </div>

                    {config.enabled && (
                      <div className="nn-architecture">
                        <div className="nn-input-group">
                          <label>Hidden Layers:</label>
                          <input
                            type="text"
                            value={config.hiddenLayers}
                            onChange={(e) => handleNnConfigChange(property, 'hiddenLayers', e.target.value)}
                            disabled={!config.enabled}
                          />
                        </div>
                        <div className="nn-input-group">
                          <label>Neurons/Layer:</label>
                          <input
                            type="text"
                            value={config.neuronsPerLayer}
                            onChange={(e) => handleNnConfigChange(property, 'neuronsPerLayer', e.target.value)}
                            disabled={!config.enabled}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Divisor */}
            <hr className="section-divider" />

            {/* Tabla de mediciones */}
            <div className="measurements-section">
              <h3>Desplazamientos Medidos</h3>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Nodo</th>
                    <th>Ux medido [m]</th>
                    <th>Uy medido [m]</th>
                  </tr>
                </thead>
                <tbody>
                  {nodes.map((node, idx) => (
                    <tr key={idx}>
                      <td>{idx}</td>
                      <td>
                        <input
                          type="text"
                          value={node.measuredUx || 0}
                          onChange={(e) => handleNodeChange(idx, 'measuredUx', e.target.value)}
                          placeholder="0.0"
                        />
                      </td>
                      <td>
                        <input
                          type="text"
                          value={node.measuredUy || 0}
                          onChange={(e) => handleNodeChange(idx, 'measuredUy', e.target.value)}
                          placeholder="0.0"
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'results' && displacements && (
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Nodo</th>
                  <th>Ux calc [m]</th>
                  <th>Uy calc [m]</th>
                  <th>|U| [m]</th>
                </tr>
              </thead>
              <tbody>
                {displacements.map((disp, idx) => {
                  const magnitude = Math.sqrt((disp.ux || 0) ** 2 + (disp.uy || 0) ** 2)
                  return (
                    <tr key={idx}>
                      <td>{disp.node}</td>
                      <td>{(disp.ux || 0).toExponential(3)}</td>
                      <td>{(disp.uy || 0).toExponential(3)}</td>
                      <td>{magnitude.toExponential(3)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default DataTable
