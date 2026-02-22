import './PropertiesPanel.css'

const PropertiesPanel = ({ selectedItem, nodes, elements, setNodes, setElements }) => {
  if (!selectedItem) {
    return (
      <div className="properties-panel">
        <h3>📋 Propiedades</h3>
        <div className="empty-state">
          <p>Selecciona un nodo o elemento para ver sus propiedades</p>
        </div>
      </div>
    )
  }

  const updateNode = (index, field, value) => {
    const updatedNodes = [...nodes]
    updatedNodes[index] = { ...updatedNodes[index], [field]: value }
    setNodes(updatedNodes)
  }

  const updateElement = (index, field, value) => {
    const updatedElements = [...elements]
    updatedElements[index] = { ...updatedElements[index], [field]: value }
    setElements(updatedElements)
  }

  if (selectedItem.type === 'node') {
    const node = nodes[selectedItem.index]
    if (!node) return null

    const bcType = node.bcType || 'free'
    const bcValue = node.bcValue !== undefined ? node.bcValue : 0

    return (
      <div className="properties-panel">
        <h3>📍 Nodo #{selectedItem.index}</h3>
        <div className="property-grid">
          <div className="property-item compact-material">
            <label>Coordenada X:</label>
            <div className="material-input-group">
              <div className="input-with-unit compact">
                <input
                  type="text"
                  value={node.x}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    if (!isNaN(val)) updateNode(selectedItem.index, 'x', val)
                  }}
                />
                <span className="unit">m</span>
              </div>
            </div>
          </div>
          <div className="property-item compact-material">
            <label>Coordenada Y:</label>
            <div className="material-input-group">
              <div className="input-with-unit compact">
                <input
                  type="text"
                  value={node.y}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    if (!isNaN(val)) updateNode(selectedItem.index, 'y', val)
                  }}
                />
                <span className="unit">m</span>
              </div>
            </div>
          </div>
        </div>

        <div className="property-section">
          <div className="property-item bc-compact">
            <label>Condición de Contorno:</label>
            <div className="bc-inline">
              <label className="checkbox-inline">
                <input
                  type="checkbox"
                  checked={bcType === 'fixed'}
                  onChange={(e) => updateNode(selectedItem.index, 'bcType', e.target.checked ? 'fixed' : 'free')}
                />
                <span>{bcType === 'fixed' ? '🔒 Fijo' : '🔓 Libre'}</span>
              </label>
              <div className="input-with-unit compact">
                <input
                  type="text"
                  value={bcValue}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    if (!isNaN(val)) updateNode(selectedItem.index, 'bcValue', val)
                  }}
                  placeholder={bcType === 'free' ? 'Fuerza (N)' : 'Despl. (m)'}
                />
                <span className="unit">{bcType === 'free' ? 'N' : 'm'}</span>
              </div>
            </div>
          </div>
        </div>

      </div>
    )
  }

  if (selectedItem.type === 'element') {
    const element = elements[selectedItem.index]
    if (!element) return null

    const node1 = nodes[element.nodes[0]]
    const node2 = nodes[element.nodes[1]]

    if (!node1 || !node2) return null

    const length = Math.sqrt(
      (node2.x - node1.x) ** 2 + (node2.y - node1.y) ** 2
    )

    // Valores por defecto para propiedades materiales
    const young = element.young !== undefined ? element.young : 210e9  // Pa (acero)
    const area = element.area !== undefined ? element.area : 0.01      // m²
    const density = element.density !== undefined ? element.density : 7850  // kg/m³

    return (
      <div className="properties-panel">
        <h3>━ Elemento #{selectedItem.index}</h3>
        <div className="property-grid">
          <div className="property-item">
            <label>Nodo Inicio:</label>
            <span>#{element.nodes[0]}</span>
          </div>
          <div className="property-item">
            <label>Nodo Fin:</label>
            <span>#{element.nodes[1]}</span>
          </div>
          <div className="property-item">
            <label>Longitud:</label>
            <span>{length.toFixed(4)} m</span>
          </div>
        </div>

        <div className="property-section">
          <h4>🔧 Propiedades del Material</h4>

          <div className="property-item compact-material">
            <label>Módulo de Young (E):</label>
            <div className="material-input-group">
              <div className="input-with-unit compact">
                <input
                  type="text"
                  value={young}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    if (!isNaN(val)) updateElement(selectedItem.index, 'young', val)
                  }}
                />
                <span className="unit">Pa</span>
              </div>
              <span className="help-subtext">({(young / 1e9).toFixed(1)} GPa)</span>
            </div>
          </div>

          <div className="property-item compact-material">
            <label>Área de sección (A):</label>
            <div className="material-input-group">
              <div className="input-with-unit compact">
                <input
                  type="text"
                  value={area}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    if (!isNaN(val)) updateElement(selectedItem.index, 'area', val)
                  }}
                />
                <span className="unit">m²</span>
              </div>
              <span className="help-subtext">({(area * 1e4).toFixed(2)} cm²)</span>
            </div>
          </div>

          <div className="property-item compact-material">
            <label>Densidad (ρ):</label>
            <div className="material-input-group">
              <div className="input-with-unit compact">
                <input
                  type="text"
                  value={density}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    if (!isNaN(val)) updateElement(selectedItem.index, 'density', val)
                  }}
                />
                <span className="unit">kg/m³</span>
              </div>
            </div>
          </div>
        </div>

        <div className="property-section">
          <h4>📊 Información Adicional</h4>
          <div className="property-item">
            <label>Rigidez axial (EA):</label>
            <span>{(young * area).toExponential(2)} N</span>
          </div>
          <div className="property-item">
            <label>Masa lineal:</label>
            <span>{(density * area).toFixed(2)} kg/m</span>
          </div>
        </div>

        <p className="help-text">
          💡 Acero: E=210 GPa, ρ=7850 kg/m³ • Aluminio: E=70 GPa, ρ=2700 kg/m³
        </p>
      </div>
    )
  }

  return null
}

export default PropertiesPanel
