import { useState } from 'react'
import './CalculateButton.css'

const CalculateButton = ({ onCalculate, isCalculating, disabled }) => {
  const [solverType, setSolverType] = useState('fem')
  const [tolerance, setTolerance] = useState(1e-6)
  const [maxIterations, setMaxIterations] = useState(50)

  const handleCalculate = () => {
    onCalculate({
      solverType,
      tolerance,
      maxIterations
    })
  }

  return (
    <div className="calculate-container">
      <button
        className="calculate-button"
        onClick={handleCalculate}
        disabled={disabled || isCalculating}
      >
        {isCalculating ? (
          <>
            <span className="spinner"></span>
            Calculando...
          </>
        ) : (
          <>
            🚀 Calcular FEM
          </>
        )}
      </button>

      <div className="calc-info">
        {disabled ? (
          <p className="warning">
            ⚠️ Necesitas al menos 2 nodos y 1 elemento
          </p>
        ) : (
          <p className="ready">
            ✓ Modelo listo para calcular
          </p>
        )}
      </div>

      <div className="solver-options">
        <h4>Configuración del Solver</h4>
        <div className="option-item">
          <label>Solver:</label>
          <select
            value={solverType}
            onChange={(e) => setSolverType(e.target.value)}
          >
            <option value="fem">FEM Clásico</option>
            <option value="pinn-gd">PINN (Gradient Descent)</option>
            <option value="pinn-nr">PINN (Newton-Raphson)</option>
          </select>
        </div>
        <div className="option-item">
          <label>Tolerancia:</label>
          <input
            type="number"
            value={tolerance}
            onChange={(e) => setTolerance(parseFloat(e.target.value))}
            step="1e-7"
          />
        </div>
        <div className="option-item">
          <label>Max Iter:</label>
          <input
            type="number"
            value={maxIterations}
            onChange={(e) => setMaxIterations(parseInt(e.target.value))}
            min="1"
            max="1000"
          />
        </div>
      </div>
    </div>
  )
}

export default CalculateButton
