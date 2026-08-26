import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import 'leaflet/dist/leaflet.css';


// Silence upstream Three.js r185 Clock deprecation warnings from third-party libraries & suppress context lost log spam
if (typeof window !== 'undefined') {
  const originalWarn = console.warn;
  console.warn = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('THREE.Clock: This module has been deprecated')
    ) {
      return;
    }
    originalWarn.apply(console, args);
  };

  const originalLog = console.log;
  console.log = (...args: any[]) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('THREE.WebGLRenderer: Context Lost')
    ) {
      return;
    }
    originalLog.apply(console, args);
  };
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
