import React, { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import { Detection } from '../../types';
import { Shield } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';

interface BathymetryViewerProps {
  detection?: Detection | null;
  depthMeters?: number;
  areaSizeMeters?: number;
  bathymetryAvailable?: boolean;
  className?: string;
}

// 3D Reconstructed Seabed Patch for a single Target
function LocalizedSeabedMesh({
  wireframe = false,
  contourMode = false,
  colorScheme = 'oceanic',
  isLight = false,
}: {
  wireframe?: boolean;
  contourMode?: boolean;
  colorScheme?: 'oceanic' | 'bathymetric' | 'thermal';
  isLight?: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  const { geometry } = useMemo(() => {
    const size = 16;
    const segments = 48;
    const geom = new THREE.PlaneGeometry(size, size, segments, segments);
    const pos = geom.attributes.position;
    const colorArray = new Float32Array(pos.count * 3);

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);

      const distFromCenter = Math.sqrt(x * x + y * y);
      const mound = Math.exp(-distFromCenter * 0.3) * 1.4;
      const ripple = Math.sin(x * 0.8) * Math.cos(y * 0.8) * 0.35;
      const elevation = mound + ripple;

      pos.setZ(i, elevation);

      const normZ = (elevation + 1) / 3; // 0 to 1
      let r = 0,
        g = 0.4,
        b = 0.8;

      if (colorScheme === 'bathymetric') {
        if (normZ < 0.25) {
          r = 0.05;
          g = 0.3;
          b = 0.9;
        } else if (normZ < 0.5) {
          r = 0.05;
          g = 0.75;
          b = 0.8;
        } else if (normZ < 0.75) {
          r = 0.9;
          g = 0.75;
          b = 0.1;
        } else {
          r = 0.95;
          g = 0.2;
          b = 0.2;
        }
      } else if (colorScheme === 'thermal') {
        r = normZ * 0.9 + 0.1;
        g = normZ * 0.4;
        b = 0.3;
      } else {
        if (isLight) {
          r = 0.4 + normZ * 0.4;
          g = 0.7 + normZ * 0.25;
          b = 0.9 + normZ * 0.1;
        } else {
          r = 0.02 + normZ * 0.1;
          g = 0.15 + normZ * 0.6;
          b = 0.35 + normZ * 0.55;
        }
      }

      colorArray[i * 3] = r;
      colorArray[i * 3 + 1] = g;
      colorArray[i * 3 + 2] = b;
    }

    geom.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
    geom.computeVertexNormals();
    return { geometry: geom, colors: colorArray };
  }, [colorScheme, isLight]);

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <mesh ref={meshRef} geometry={geometry} receiveShadow>
        <meshStandardMaterial
          vertexColors
          roughness={0.75}
          metalness={0.15}
          wireframe={wireframe}
        />
      </mesh>

      {contourMode && (
        <mesh geometry={geometry} position={[0, 0, 0.01]}>
          <meshBasicMaterial
            color={isLight ? '#0284c7' : '#38bdf8'}
            wireframe
            opacity={isLight ? 0.45 : 0.35}
            transparent
          />
        </mesh>
      )}
    </group>
  );
}

// 3D Model of the Subsea Anomaly Target
function TargetReconstructionObject({
  detection,
  isLight = false,
}: {
  detection?: Detection | null;
  isLight?: boolean;
}) {
  const targetClass = detection?.class || 'ghost_gear';

  let color = isLight ? '#0284c7' : '#22d3ee';
  if (targetClass === 'ghost_gear') color = '#f59e0b';
  if (targetClass === 'shipwreck') color = '#ec4899';
  if (targetClass === 'unexploded_ordnance') color = '#ef4444';
  if (targetClass === 'pipeline_anomaly') color = '#8b5cf6';

  return (
    <group position={[0, 1.4, 0]}>
      {targetClass === 'shipwreck' ? (
        <group>
          <mesh position={[0, 0, 0]} rotation={[0, 0.4, 0]}>
            <boxGeometry args={[1.6, 0.8, 3.8]} />
            <meshStandardMaterial color={color} metalness={0.8} roughness={0.3} wireframe={false} />
          </mesh>
          <mesh position={[0, 0.6, -0.4]} rotation={[0, 0.4, 0]}>
            <boxGeometry args={[1.2, 0.7, 1.4]} />
            <meshStandardMaterial color={color} wireframe={false} />
          </mesh>
        </group>
      ) : targetClass === 'pipeline_anomaly' ? (
        <group rotation={[0, 0, Math.PI / 2]}>
          <mesh position={[0, 0, 0]}>
            <cylinderGeometry args={[0.35, 0.35, 6, 24]} />
            <meshStandardMaterial color={color} metalness={0.9} roughness={0.2} />
          </mesh>
        </group>
      ) : targetClass === 'unexploded_ordnance' ? (
        <group rotation={[0.2, 0.3, 0]}>
          <mesh>
            <cylinderGeometry args={[0.3, 0.35, 1.8, 16]} />
            <meshStandardMaterial color={color} metalness={0.9} />
          </mesh>
          <mesh position={[0, 1.0, 0]}>
            <coneGeometry args={[0.3, 0.5, 16]} />
            <meshStandardMaterial color="#ef4444" />
          </mesh>
        </group>
      ) : (
        <group>
          <mesh>
            <dodecahedronGeometry args={[0.85, 1]} />
            <meshStandardMaterial color={color} wireframe opacity={0.85} transparent />
          </mesh>
          <mesh>
            <icosahedronGeometry args={[0.6, 0]} />
            <meshStandardMaterial color="#f59e0b" roughness={0.4} />
          </mesh>
        </group>
      )}

      <mesh position={[0, 1.5, 0]}>
        <octahedronGeometry args={[0.3, 0]} />
        <meshBasicMaterial color={color} wireframe />
      </mesh>

      <mesh position={[1.4, -1.38, 1.4]} rotation={[-Math.PI / 2, 0, 0.4]}>
        <planeGeometry args={[1.8, 4.2]} />
        <meshBasicMaterial color={isLight ? '#94a3b8' : '#020712'} opacity={0.85} transparent />
      </mesh>

      <Html position={[1.2, 0.8, 0]} center>
        <div className="bg-[#040D1B]/90 dark:bg-[#040D1B]/90 light:bg-white/95 backdrop-blur-md border border-cyan-400/50 dark:border-cyan-400/50 light:border-sky-300 px-2 py-0.5 rounded text-[10px] font-mono text-cyan-300 dark:text-cyan-300 light:text-sky-800 shadow-xl whitespace-nowrap pointer-events-none font-bold">
          L: {detection?.acousticShadow?.lengthMeters || 3.4}m | H:{' '}
          {detection?.acousticShadow?.estimatedHeightMeters || 1.2}m
        </div>
      </Html>
    </group>
  );
}

export const BathymetryViewer: React.FC<BathymetryViewerProps> = ({
  detection,
  depthMeters = 42.5,
  areaSizeMeters = 25,
  bathymetryAvailable = true,
  className = 'h-[360px] w-full',
}) => {
  const { isLight } = useTheme();
  const [wireframe, setWireframe] = useState(false);
  const [contourMode, setContourMode] = useState(true);
  const [colorScheme, setColorScheme] = useState<'oceanic' | 'bathymetric' | 'thermal'>('oceanic');

  if (!bathymetryAvailable) {
    return (
      <div
        className={`rounded-xl bg-[#040D1B]/80 dark:bg-[#040D1B]/80 light:bg-white/90 backdrop-blur-xl border border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200 flex flex-col items-center justify-center p-8 text-center font-mono ${className}`}
      >
        <Shield className="w-10 h-10 text-slate-600 dark:text-slate-600 light:text-slate-400 mb-3" />
        <div className="text-sm font-bold text-slate-300 dark:text-slate-300 light:text-slate-800 uppercase tracking-wider">
          Bathymetry unavailable for this mission.
        </div>
        <p className="text-xs text-slate-500 dark:text-slate-500 light:text-slate-600 mt-1 max-w-sm">
          Multi-beam bathymetry digital elevation model (DEM) is required to reconstruct 3D seabed
          morphology.
        </p>
      </div>
    );
  }

  const bgColor = isLight ? '#d2ecf9' : '#030914';

  return (
    <div
      className={`relative rounded-xl overflow-hidden bg-[#030914] dark:bg-[#030914] light:bg-[#d2ecf9] border border-cyan-500/20 dark:border-cyan-500/20 light:border-sky-300 shadow-[0_8px_32px_rgba(0,0,0,0.6)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.6)] light:shadow-md transition-colors ${className}`}
    >
      {/* Floating HUD Controls */}
      <div className="absolute top-3 left-3 z-10 flex items-center gap-2 bg-[#040D1B]/85 dark:bg-[#040D1B]/85 light:bg-white/95 backdrop-blur-md border border-cyan-500/30 dark:border-cyan-500/30 light:border-sky-200 px-3 py-1.5 rounded-lg text-xs font-mono shadow-xl text-slate-200 dark:text-slate-200 light:text-slate-800">
        <span className="w-2 h-2 rounded-full bg-cyan-400 dark:bg-cyan-400 light:bg-sky-600 animate-pulse" />
        <span className="text-white dark:text-white light:text-slate-900 font-bold uppercase text-[11px] tracking-wider">
          3D SEABED RECONSTRUCTION
        </span>
        <span className="text-slate-500 dark:text-slate-500 light:text-slate-300">|</span>
        <span className="text-cyan-400 dark:text-cyan-400 light:text-sky-700 font-bold">
          {depthMeters.toFixed(1)}m DEPTH
        </span>
      </div>

      {/* Layer Options */}
      <div className="absolute top-3 right-3 z-10 flex items-center gap-1 bg-[#040D1B]/85 dark:bg-[#040D1B]/85 light:bg-white/95 backdrop-blur-md border border-cyan-500/30 dark:border-cyan-500/30 light:border-sky-200 p-1 rounded-lg text-[10px] font-mono shadow-xl">
        <button
          onClick={() => setContourMode(!contourMode)}
          className={`px-2 py-1 rounded transition-all font-bold ${
            contourMode
              ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-sky-800 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-300'
              : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900'
          }`}
          title="Toggle Depth Contours"
        >
          CONTOURS
        </button>
        <button
          onClick={() => setWireframe(!wireframe)}
          className={`px-2 py-1 rounded transition-all font-bold ${
            wireframe
              ? 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-sky-800 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-300'
              : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-white dark:hover:text-white light:hover:text-slate-900'
          }`}
          title="Toggle Wireframe Mesh"
        >
          WIREFRAME
        </button>
        <select
          value={colorScheme}
          onChange={(e) => setColorScheme(e.target.value as any)}
          className="bg-[#020712] dark:bg-[#020712] light:bg-slate-100 border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-300 rounded px-1.5 py-0.5 text-cyan-300 dark:text-cyan-300 light:text-sky-800 text-[10px] font-bold focus:outline-none"
        >
          <option value="oceanic">Oceanic</option>
          <option value="bathymetric">Bathymetric DEM</option>
          <option value="thermal">Thermal SNR</option>
        </select>
      </div>

      {/* Canvas */}
      <Canvas
        camera={{ position: [6, 7, 9], fov: 42 }}
        gl={{
          antialias: true,
          powerPreference: 'default',
          preserveDrawingBuffer: false,
        }}
        onCreated={({ gl }) => {
          gl.domElement.addEventListener('webglcontextlost', (e) => {
            e.preventDefault();
          });
        }}
      >
        <color attach="background" args={[bgColor]} />
        <fog attach="fog" args={[bgColor, isLight ? 8 : 6, isLight ? 28 : 24]} />

        <ambientLight intensity={isLight ? 0.7 : 0.4} />
        <directionalLight
          position={[8, 14, 6]}
          intensity={isLight ? 1.0 : 0.8}
          color={isLight ? '#ffffff' : '#38bdf8'}
        />
        <pointLight
          position={[0, 4, 0]}
          intensity={isLight ? 1.8 : 1.5}
          color={isLight ? '#0284c7' : '#00f0ff'}
          distance={15}
        />

        <LocalizedSeabedMesh
          wireframe={wireframe}
          contourMode={contourMode}
          colorScheme={colorScheme}
          isLight={isLight}
        />

        <TargetReconstructionObject detection={detection} isLight={isLight} />

        <OrbitControls
          maxPolarAngle={Math.PI / 2 - 0.05}
          minDistance={3}
          maxDistance={18}
          enableDamping
          dampingFactor={0.06}
        />
      </Canvas>
    </div>
  );
};
