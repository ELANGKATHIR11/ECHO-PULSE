/**
 * ==============================================================================
 * EchoPulseNet Marine Sonar Intelligence Platform
 * Electron Desktop Application Shell
 * Multi-Silicon Hardware Accelerated Native Desktop Runtime
 * Supports NVIDIA RTX 5060 DGPU + Intel(R) AI Boost NPU Co-Processing
 * ==============================================================================
 */

import { app, BrowserWindow, shell, ipcMain } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn, exec } from 'child_process';
import http from 'http';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Enable Chromium High-Performance Hardware GPU Acceleration for RTX 5060 & WebGL2
app.commandLine.appendSwitch('enable-gpu-rasterization');
app.commandLine.appendSwitch('enable-zero-copy');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('enable-native-gpu-memory-buffers');
app.commandLine.appendSwitch('high-dpi-support', '1');
app.commandLine.appendSwitch('disable-http-cache');

let mainWindow = null;
let backendProcess = null;
const SERVER_PORT = 8000;
const SERVER_URL = `http://127.0.0.1:${SERVER_PORT}`;

// 1. Ensure PostgreSQL service is active
function ensurePostgres() {
  return new Promise((resolve) => {
    console.log('[EchoPulseNet Desktop] Checking PostgreSQL Database Service...');
    if (process.platform === 'win32') {
      exec('net start postgresql-x64-18', (err, stdout, stderr) => {
        if (err) {
          // If already running or manual, attempt pg_ctl directly
          exec('& "F:\\Program Files\\PostgreSQL\\18\\bin\\pg_ctl.exe" start -D "F:\\Program Files\\PostgreSQL\\18\\data" -w', (pgErr) => {
            resolve(true);
          });
        } else {
          console.log('[EchoPulseNet Desktop] PostgreSQL Service Active.');
          resolve(true);
        }
      });
    } else {
      resolve(true);
    }
  });
}

// 2. Start Unified Python AI + Frontend Backend
function startBackend() {
  console.log('[EchoPulseNet Desktop] Launching Unified AI & Web Server on port 8000...');
  
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
  
  backendProcess = spawn(pythonCmd, ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', String(SERVER_PORT)], {
    cwd: path.join(__dirname, 'backend'),
    env: { ...process.env, PYTHONUNBUFFERED: '1', OPENBLAS_NUM_THREADS: '1', OMP_NUM_THREADS: '1' },
    stdio: 'inherit'
  });

  backendProcess.on('error', (err) => {
    console.error('[EchoPulseNet Desktop] Failed to start backend process:', err);
  });

  backendProcess.on('exit', (code, signal) => {
    console.log(`[EchoPulseNet Desktop] Backend process exited with code ${code} signal ${signal}`);
  });
}

function waitForServer(url, timeout = 25000) {
  const start = Date.now();
  return new Promise((resolve) => {
    const check = () => {
      http.get(`${url}/api/v1/system/telemetry`, (res) => {
        if (res.statusCode === 200) {
          console.log('[EchoPulseNet Desktop] Backend server online & responding.');
          resolve(true);
        } else {
          retry();
        }
      }).on('error', () => {
        retry();
      });
    };

    const retry = () => {
      if (Date.now() - start > timeout) {
        console.warn('[EchoPulseNet Desktop] Server health check timed out, loading UI directly...');
        resolve(false);
      } else {
        setTimeout(check, 350);
      }
    };

    check();
  });
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1540,
    height: 980,
    minWidth: 1100,
    minHeight: 740,
    backgroundColor: '#020712',
    title: 'EchoPulseNet PRO | Marine Sonar Intelligence Platform (Native Desktop Edition)',
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webgl: true,
      backgroundThrottling: false,
    },
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // Load from Unified Single Server
  await mainWindow.loadURL(SERVER_URL);
  mainWindow.show();

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  await ensurePostgres();
  startBackend();
  await waitForServer(SERVER_URL);
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (backendProcess) {
    console.log('[EchoPulseNet Desktop] Terminating embedded backend process...');
    backendProcess.kill();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
