import express from 'express';
import { solveFEM, solvePINN, solveGeneric } from '../services/pythonRunner.js';

const router = express.Router();

/**
 * POST /api/fem/solve
 * Resolver modelo FEM clásico
 * 
 * Body:
 * {
 *   nodes: [{x, y, fixed}, ...],
 *   elements: [{nodes: [i, j]}, ...],
 *   material: {young, area, density},
 *   loads: [force_node0, force_node1, ...],
 *   fixed_dofs: [0, 3, ...],
 *   solver_config: {tolerance, max_iterations, n_increments}
 * }
 */
router.post('/solve', async (req, res) => {
  try {
    console.log('📥 Received FEM solve request');
    console.log('   Nodes:', req.body.nodes?.length || 0);
    console.log('   Elements:', req.body.elements?.length || 0);

    const result = await solveFEM(req.body);

    console.log('✅ FEM solve completed successfully');
    res.json({
      success: true,
      result: result
    });

  } catch (error) {
    console.error('❌ Error solving FEM:', error.message);
    res.status(500).json({
      success: false,
      error: error.message,
      details: error.stderr || error.stdout
    });
  }
});

/**
 * POST /api/fem/solve-pinn
 * Resolver problema inverso con PINN
 * 
 * Body:
 * {
 *   nodes: [{x, y, fixed}, ...],
 *   elements: [{nodes: [i, j]}, ...],
 *   material: {young, area, density},
 *   loads: [...],
 *   measured_disp: [...],
 *   measured_dofs: [...],
 *   solver_type: 'gradient_descent' | 'newton_raphson',
 *   solver_config: {...}
 * }
 */
router.post('/solve-pinn', async (req, res) => {
  try {
    console.log('📥 Received PINN solve request');
    console.log('   Solver type:', req.body.solver_type || 'gradient_descent');
    console.log('   Nodes:', req.body.nodes?.length || 0);
    console.log('   Measurements:', req.body.measured_dofs?.length || 0);

    const result = await solvePINN(req.body);

    console.log('✅ PINN solve completed successfully');
    res.json({
      success: true,
      result: result
    });

  } catch (error) {
    console.error('❌ Error solving PINN:', error.message);
    res.status(500).json({
      success: false,
      error: error.message,
      details: error.stderr || error.stdout
    });
  }
});

/**
 * POST /api/fem/solve-generic
 * Resolver problema con solver genérico (FEM o PINN según configuración)
 * 
 * Body:
 * {
 *   nodes: [{x, y, fixed, measured_ux, measured_uy}, ...],
 *   elements: [{nodes: [i, j]}, ...],
 *   material: {young, area, density},
 *   loads: [...],
 *   nn_config: {young: {enabled, hiddenLayers, neuronsPerLayer}, ...},
 *   solver_type: 'fem' | 'pinn-gd' | 'pinn-nr',
 *   solver_config: {...}
 * }
 */
router.post('/solve-generic', async (req, res) => {
  try {
    console.log('📥 Received GENERIC solve request');
    console.log('   Solver type:', req.body.solver_type || 'fem');
    console.log('   Nodes:', req.body.nodes?.length || 0);
    console.log('   Elements:', req.body.elements?.length || 0);

    // Log NN config if present
    const nnConfig = req.body.nn_config || {};
    const nnProps = Object.keys(nnConfig).filter(k => nnConfig[k]?.enabled);
    if (nnProps.length > 0) {
      console.log('   NN Properties:', nnProps.join(', '));
    }

    const result = await solveGeneric(req.body);

    console.log('✅ GENERIC solve completed successfully');
    res.json({
      success: true,
      result: result
    });

  } catch (error) {
    console.error('❌ Error solving with generic solver:', error.message);
    res.status(500).json({
      success: false,
      error: error.message,
      details: error.stderr || error.stdout
    });
  }
});

/**
 * GET /api/fem/info
 * Información sobre el solver disponible
 */
router.get('/info', (req, res) => {
  res.json({
    version: '1.0.0',
    solvers: {
      fem: {
        name: 'FEM Clásico',
        description: 'Solver de elementos finitos estándar',
        element_types: ['truss1d', 'truss2d'],
        methods: ['incremental_newton_raphson']
      },
      pinn: {
        name: 'Physics-Informed Neural Networks',
        description: 'Identificación de parámetros con redes neuronales',
        methods: ['gradient_descent', 'newton_raphson']
      }
    },
    python_version: 'Python 3.12',
    dependencies: ['numpy', 'torch', 'matplotlib']
  });
});

export default router;
