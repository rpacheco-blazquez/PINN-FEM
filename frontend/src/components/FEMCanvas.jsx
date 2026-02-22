import { useRef, useEffect, useState } from 'react'
import './FEMCanvas.css'

const FEMCanvas = ({ nodes, setNodes, elements, setElements, displacements, selectedItem, setSelectedItem }) => {
  const canvasRef = useRef(null)
  const [mode, setMode] = useState('select') // 'select', 'node' or 'element'
  const [scale] = useState(50) // Escala de visualización
  const [offset] = useState({ x: 50, y: 50 }) // Offset para centrar
  const [deformationScale, setDeformationScale] = useState(100) // Escala de deformación

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height

    // Limpiar canvas
    ctx.fillStyle = '#2a2a2a'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Dibujar grid
    drawGrid(ctx, canvas.width, canvas.height)

    // Dibujar elementos (barras)
    elements.forEach((elem, idx) => {
      const node1 = nodes[elem.nodes[0]]
      const node2 = nodes[elem.nodes[1]]

      if (!node1 || !node2) return

      const isSelected = selectedItem?.type === 'element' && selectedItem?.index === idx

      drawElement(ctx, node1, node2, displacements, deformationScale, isSelected)
    })

    // Dibujar nodos
    nodes.forEach((node, idx) => {
      const isSelected = selectedItem?.type === 'node' && selectedItem?.index === idx
      drawNode(ctx, node, idx, displacements, deformationScale, isSelected)
    })

  }, [nodes, elements, displacements, selectedItem, scale, offset, deformationScale])

  const drawGrid = (ctx, width, height) => {
    ctx.strokeStyle = '#3a3a3a'
    ctx.lineWidth = 1

    const gridSize = 50
    for (let x = 0; x < width; x += gridSize) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }

    for (let y = 0; y < height; y += gridSize) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }
  }

  const drawNode = (ctx, node, idx, disp, dispScale, isSelected) => {
    let x = offset.x + node.x * scale
    let y = offset.y + node.y * scale

    // Aplicar deformación si existe
    if (disp && disp[idx]) {
      x += (disp[idx].ux || 0) * dispScale
      y += (disp[idx].uy || 0) * dispScale
    }

    // Dibujar nodo
    ctx.fillStyle = isSelected ? '#ffd700' : '#4a9eff'
    ctx.beginPath()
    ctx.arc(x, y, isSelected ? 8 : 6, 0, 2 * Math.PI)
    ctx.fill()

    // Dibujar índice
    ctx.fillStyle = '#e0e0e0'
    ctx.font = '12px monospace'
    ctx.fillText(`N${idx}`, x + 10, y - 10)

    // Dibujar soporte si está fijo
    const isFixed = node.fixed || node.bcType === 'fixed'
    if (isFixed) {
      ctx.strokeStyle = '#ff4444'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(x - 10, y + 8)
      ctx.lineTo(x + 10, y + 8)
      ctx.stroke()

      // Triángulos de soporte
      for (let i = -10; i <= 10; i += 5) {
        ctx.beginPath()
        ctx.moveTo(x + i, y + 8)
        ctx.lineTo(x + i - 3, y + 13)
        ctx.stroke()
      }
    }
  }

  const drawElement = (ctx, node1, node2, disp, dispScale, isSelected) => {
    let x1 = offset.x + node1.x * scale
    let y1 = offset.y + node1.y * scale
    let x2 = offset.x + node2.x * scale
    let y2 = offset.y + node2.y * scale

    // Aplicar deformación si existe
    if (disp) {
      const idx1 = nodes.findIndex(n => n === node1)
      const idx2 = nodes.findIndex(n => n === node2)

      if (disp[idx1]) {
        x1 += (disp[idx1].ux || 0) * dispScale
        y1 += (disp[idx1].uy || 0) * dispScale
      }
      if (disp[idx2]) {
        x2 += (disp[idx2].ux || 0) * dispScale
        y2 += (disp[idx2].uy || 0) * dispScale
      }
    }

    // Dibujar elemento original (tenue si hay deformación)
    if (disp) {
      ctx.strokeStyle = '#555555'
      ctx.lineWidth = 1
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.moveTo(offset.x + node1.x * scale, offset.y + node1.y * scale)
      ctx.lineTo(offset.x + node2.x * scale, offset.y + node2.y * scale)
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Dibujar elemento (deformado si hay desplazamientos)
    ctx.strokeStyle = isSelected ? '#ffd700' : (disp ? '#00ff88' : '#888888')
    ctx.lineWidth = isSelected ? 4 : 3
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()
  }

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    if (mode === 'select') {
      // Modo selección: detectar clic en nodo o elemento
      const closestNodeIdx = findClosestNode(x, y)

      if (closestNodeIdx !== -1) {
        setSelectedItem({ type: 'node', index: closestNodeIdx })
      } else {
        // Buscar si hizo clic cerca de un elemento
        const closestElementIdx = findClosestElement(x, y)
        if (closestElementIdx !== -1) {
          setSelectedItem({ type: 'element', index: closestElementIdx })
        } else {
          setSelectedItem(null)
        }
      }
    } else if (mode === 'node') {
      // Añadir nuevo nodo
      const nodeX = (x - offset.x) / scale
      const nodeY = (y - offset.y) / scale

      setNodes([...nodes, {
        x: nodeX,
        y: nodeY,
        fixed: false,
        bcType: 'free',
        bcValue: 0
      }])
    } else if (mode === 'element') {
      // Seleccionar nodo más cercano para crear elemento
      const closestNodeIdx = findClosestNode(x, y)

      if (closestNodeIdx !== -1) {
        if (selectedItem?.type === 'node' && selectedItem?.index !== closestNodeIdx) {
          // Crear elemento entre el nodo previamente seleccionado y este
          const newElement = {
            nodes: [selectedItem.index, closestNodeIdx],
            young: 210e9,    // Pa (acero por defecto)
            area: 0.01,      // m² por defecto
            density: 7850    // kg/m³ (acero)
          }
          setElements([...elements, newElement])
          setSelectedItem(null)
        } else {
          setSelectedItem({ type: 'node', index: closestNodeIdx })
        }
      }
    }
  }

  const findClosestNode = (x, y) => {
    let minDist = Infinity
    let closestIdx = -1

    nodes.forEach((node, idx) => {
      const nodeX = offset.x + node.x * scale
      const nodeY = offset.y + node.y * scale
      const dist = Math.sqrt((x - nodeX) ** 2 + (y - nodeY) ** 2)

      if (dist < minDist && dist < 20) {
        minDist = dist
        closestIdx = idx
      }
    })

    return closestIdx
  }

  const findClosestElement = (x, y) => {
    let minDist = Infinity
    let closestIdx = -1

    elements.forEach((elem, idx) => {
      const node1 = nodes[elem.nodes[0]]
      const node2 = nodes[elem.nodes[1]]

      if (!node1 || !node2) return

      const x1 = offset.x + node1.x * scale
      const y1 = offset.y + node1.y * scale
      const x2 = offset.x + node2.x * scale
      const y2 = offset.y + node2.y * scale

      // Distancia punto a línea
      const dist = distanceToSegment(x, y, x1, y1, x2, y2)

      if (dist < minDist && dist < 10) {
        minDist = dist
        closestIdx = idx
      }
    })

    return closestIdx
  }

  const distanceToSegment = (px, py, x1, y1, x2, y2) => {
    const dx = x2 - x1
    const dy = y2 - y1
    const lenSq = dx * dx + dy * dy

    if (lenSq === 0) return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    let t = ((px - x1) * dx + (py - y1) * dy) / lenSq
    t = Math.max(0, Math.min(1, t))

    const projX = x1 + t * dx
    const projY = y1 + t * dy

    return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2)
  }

  return (
    <div className="fem-canvas-container">
      <div className="canvas-toolbar">
        <button
          className={mode === 'select' ? 'active' : ''}
          onClick={() => setMode('select')}
          title="Modo selección (clic para seleccionar)"
        >
          ➤ Selección
        </button>
        <button
          className={mode === 'node' ? 'active' : ''}
          onClick={() => setMode(mode === 'node' ? 'select' : 'node')}
          title="Añadir nodos (clic de nuevo para deseleccionar)"
        >
          ⬤ Nodo
        </button>
        <button
          className={mode === 'element' ? 'active' : ''}
          onClick={() => setMode(mode === 'element' ? 'select' : 'element')}
          title="Conectar nodos (clic de nuevo para deseleccionar)"
        >
          ─ Elemento
        </button>
        <div className="separator"></div>
        <button onClick={() => { setNodes([]); setElements([]); setSelectedItem(null) }} title="Limpiar todo">
          🗑️ Limpiar
        </button>
        {displacements && (
          <>
            <div className="separator"></div>
            <label>
              Escala deformación:
              <input
                type="range"
                min="1"
                max="1000"
                value={deformationScale}
                onChange={(e) => setDeformationScale(Number(e.target.value))}
              />
              <span>{deformationScale}x</span>
            </label>
          </>
        )}
      </div>
      <canvas
        ref={canvasRef}
        className="fem-canvas"
        onClick={handleCanvasClick}
      />
      <div className="canvas-info">
        <span>Modo: {
          mode === 'select' ? '➤ Selección' :
            mode === 'node' ? '⬤ Añadir Nodos' :
              '─ Crear Elementos'
        }</span>
        <span>|</span>
        <span>Nodos: {nodes.length}</span>
        <span>|</span>
        <span>Elementos: {elements.length}</span>
        {displacements && (
          <>
            <span>|</span>
            <span style={{ color: '#00ff88' }}>✓ Solución calculada</span>
          </>
        )}
      </div>
    </div>
  )
}

export default FEMCanvas
