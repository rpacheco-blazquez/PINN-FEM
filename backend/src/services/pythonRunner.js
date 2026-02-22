import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get directory name in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Path to Python scripts
const PYTHON_DIR = path.join(__dirname, '../../../FEM/python');

/**
 * Ejecuta el solver FEM clásico
 * @param {Object} problemData - Datos del problema (nodes, elements, material, etc.)
 * @returns {Promise<Object>} - Resultado del solver (displacements, stresses, etc.)
 */
export async function solveFEM(problemData) {
  const scriptPath = path.join(PYTHON_DIR, 'api_fem_solver.py');

  return await runPythonScript(scriptPath, problemData, {
    timeout: 60000 // 60 segundos
  });
}

/**
 * Ejecuta el solver PINN (problema inverso)
 * @param {Object} problemData - Datos del problema con measurements
 * @returns {Promise<Object>} - Resultado con parámetros identificados
 */
export async function solvePINN(problemData) {
  const solverType = problemData.solver_type || 'gradient_descent';
  const scriptPath = path.join(PYTHON_DIR, `api_pinn_${solverType}.py`);

  return await runPythonScript(scriptPath, problemData, {
    timeout: 300000 // 5 minutos para PINN
  });
}

/**
 * Ejecuta el solver genérico (puede ser FEM o PINN según configuración)
 * @param {Object} problemData - Datos del problema completo
 * @returns {Promise<Object>} - Resultado del solver
 */
export async function solveGeneric(problemData) {
  const scriptPath = path.join(PYTHON_DIR, 'examples', 'generic.py');
  const solverType = problemData.solver_type || 'fem';

  // Determinar timeout según tipo de solver
  const timeout = solverType.startsWith('pinn') ? 300000 : 60000;

  return await runPythonScript(scriptPath, problemData, {
    timeout,
    cwd: path.join(PYTHON_DIR, 'examples')
  });
}

/**
 * Ejecuta un script de Python y devuelve el resultado parseado
 * @param {string} scriptPath - Ruta al script Python
 * @param {Object} inputData - Datos de entrada para el script
 * @param {Object} options - Opciones (timeout, cwd, etc.)
 * @returns {Promise<Object>} - Resultado parseado del script
 */
async function runPythonScript(scriptPath, inputData, options = {}) {
  const {
    timeout = 60000,
    cwd = PYTHON_DIR
  } = options;

  // Create temporary files for input/output
  const tempDir = path.join(__dirname, '../../../temp');
  await fs.mkdir(tempDir, { recursive: true });

  const timestamp = Date.now();
  const inputFile = path.join(tempDir, `input_${timestamp}.json`);
  const outputFile = path.join(tempDir, `output_${timestamp}.json`);

  try {
    // Write input data to temp file
    await fs.writeFile(inputFile, JSON.stringify(inputData, null, 2));

    // Execute Python script
    const result = await executePython(scriptPath, [inputFile, outputFile], {
      cwd,
      timeout
    });

    // Read output file
    const outputData = await fs.readFile(outputFile, 'utf-8');
    const parsedResult = JSON.parse(outputData);

    // Clean up temp files
    await fs.unlink(inputFile).catch(() => { });
    await fs.unlink(outputFile).catch(() => { });

    return parsedResult;

  } catch (error) {
    // Clean up temp files on error
    await fs.unlink(inputFile).catch(() => { });
    await fs.unlink(outputFile).catch(() => { });

    throw error;
  }
}

/**
 * Ejecuta Python con spawn y maneja timeout
 * @param {string} scriptPath - Ruta al script
 * @param {Array<string>} args - Argumentos
 * @param {Object} options - Opciones de spawn
 * @returns {Promise<Object>} - {stdout, stderr, exitCode}
 */
function executePython(scriptPath, args, options) {
  return new Promise((resolve, reject) => {
    const { timeout = 60000, cwd } = options;

    console.log(`🐍 Executing: python "${scriptPath}" ${args.join(' ')}`);
    console.log(`   Working dir: ${cwd}`);

    const python = spawn('python', [scriptPath, ...args], {
      cwd,
      shell: true
    });

    let stdout = '';
    let stderr = '';
    let timedOut = false;

    // Set timeout
    const timer = setTimeout(() => {
      timedOut = true;
      python.kill('SIGTERM');
      reject(new Error(`Python script timed out after ${timeout}ms`));
    }, timeout);

    // Collect stdout
    python.stdout.on('data', (data) => {
      const text = data.toString();
      stdout += text;
      console.log(`   [Python stdout]: ${text.trim()}`);
    });

    // Collect stderr
    python.stderr.on('data', (data) => {
      const text = data.toString();
      stderr += text;
      console.error(`   [Python stderr]: ${text.trim()}`);
    });

    // Handle exit
    python.on('close', (code) => {
      clearTimeout(timer);

      if (timedOut) {
        return; // Already rejected
      }

      if (code !== 0) {
        const error = new Error(`Python script failed with exit code ${code}`);
        error.stdout = stdout;
        error.stderr = stderr;
        error.exitCode = code;
        reject(error);
      } else {
        resolve({ stdout, stderr, exitCode: code });
      }
    });

    // Handle spawn errors
    python.on('error', (error) => {
      clearTimeout(timer);
      reject(new Error(`Failed to spawn Python: ${error.message}`));
    });
  });
}

export default {
  solveFEM,
  solvePINN,
  solveGeneric
};
