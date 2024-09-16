const { exec, execFile } = require('child_process');
const path = require('path');
const os = require('os');
document.getElementById('search').addEventListener('click', () => {
  const query = document.getElementById('query').value;
  const indicator = document.getElementById('indicator');
  const folderInput = document.getElementById('folderInput');
  const output = document.getElementById('output');
  let modelVersion = '';
  if (folderInput.files.length === 0) {
    alert("Please select a folder first.");
    return;
  }
  if (query === '') {
    alert('Please enter a query');
    return;
  }
  const pythonCmd = os.platform() === 'win32' ? 'python' : 'python3';
  const folderPath = path.dirname(folderInput.files[0].path);
  const embeddingsModelPath = path.join(__dirname, '/extraResources/all-MiniLM-L6-v2.gguf2.f16.gguf');
  const scriptPath = path.join(__dirname, '/extraResources/index.py');
  console.log('Script Path:', scriptPath);
  console.log('Embeddings Model Path:', embeddingsModelPath);
  exec(`${pythonCmd} --version`, (error, stdout, stderr) => {
    if (error) {
      alert('Python is not installed or not found in PATH. Please install Python and try again.');
      console.error(`Error: ${error.message}`);
      return;
    }

    console.log(`Python version: ${stdout || stderr}`);

    const ollamaCmd = 'ollama list';
    exec(ollamaCmd, (error, stdout) => {
      if (error) {
        console.error('Ollama is not installed or failed to execute:', error);
        document.getElementById('ollamaVersion').innerHTML = 'Ollama not installed or no models found';
        return;
      }
      const modelList = stdout.trim();
      console.log(`Ollama models:\n${modelList}`);

      if (modelList) {
        const models = modelList.split('\n').slice(1);
        const firstModel = models[0].split('\t')[0];
        modelVersion = firstModel.trim();
        document.getElementById('ollamaVersion').innerHTML = `Using model: ${modelVersion}`;
      } 
      else {
        alert('No models found in Ollama. Please add models before proceeding.');
        return;
      }
      indicator.style.display = 'block';
      console.log(`Executing: ${pythonCmd} ${scriptPath} ${modelVersion}`);
      execFile(pythonCmd, [scriptPath, query, folderPath, modelVersion, embeddingsModelPath], (error, stdout, stderr) => {
        if (stderr) {
          console.error(`Stderr: ${stderr}`);
        }
        if (error) {
          console.error(`Error: ${error.message}`);
          indicator.style.display = 'none';
          return;
        }
        if (stdout) {
          output.textContent = `Response:\n${stdout}`;
        } else {
          output.textContent = 'No response received from the Python script.';
        }
        indicator.style.display = 'none';
      });
    });
  });
});