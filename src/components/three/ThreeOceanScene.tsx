import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';
import { Detection, Mission, RenderProfile } from '../../types';
import { useTheme } from '../../context/ThemeContext';

interface ThreeOceanSceneProps {
  mission?: Mission;
  detections?: Detection[];
  selectedDetectionId?: string | null;
  onSelectDetection?: (d: Detection) => void;
  renderProfile?: RenderProfile;
}

// Low-poly Subsea AUV / Towfish
function SonarVehicle({
  position = [0, 2.5, 0],
  isLight = false,
}: {
  position?: [number, number, number];
  isLight?: boolean;
}) {
  const vehicleRef = useRef<THREE.Group>(null);
  const fanRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (vehicleRef.current) {
      vehicleRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 0.8) * 0.15;
      vehicleRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.5) * 0.04;
      vehicleRef.current.rotation.x = Math.cos(state.clock.elapsedTime * 0.6) * 0.03;
    }
    if (fanRef.current) {
      const scale = (Math.sin(state.clock.elapsedTime * 2.5) + 1) * 0.5;
      fanRef.current.scale.set(1 + scale * 0.2, 1, 1 + scale * 0.2);
    }
  });

  return (
    <group ref={vehicleRef} position={position}>
      {/* Torpedo hull */}
      <mesh castShadow>
        <cylinderGeometry args={[0.3, 0.35, 2.2, 16]} />
        <meshStandardMaterial
          color={isLight ? '#0284c7' : '#0284c7'}
          metalness={0.8}
          roughness={0.2}
        />
      </mesh>

      {/* Nose cone */}
      <mesh position={[0, 1.25, 0]}>
        <coneGeometry args={[0.3, 0.5, 16]} />
        <meshStandardMaterial
          color={isLight ? '#0369a1' : '#00f0ff'}
          emissive={isLight ? '#0284c7' : '#00f0ff'}
          emissiveIntensity={0.3}
          metalness={0.9}
        />
      </mesh>

      {/* Acoustic Side-scan Transducers */}
      <mesh position={[0.4, 0, 0]}>
        <boxGeometry args={[0.15, 1.2, 0.15]} />
        <meshStandardMaterial color="#f59e0b" metalness={0.9} roughness={0.1} />
      </mesh>
      <mesh position={[-0.4, 0, 0]}>
        <boxGeometry args={[0.15, 1.2, 0.15]} />
        <meshStandardMaterial color="#f59e0b" metalness={0.9} roughness={0.1} />
      </mesh>

      {/* Stabilizer Fins */}
      <mesh position={[0, -0.9, 0]}>
        <boxGeometry args={[1.2, 0.1, 0.3]} />
        <meshStandardMaterial color={isLight ? '#334155' : '#0f172a'} />
      </mesh>

      {/* Sonar Acoustic Fan Sweep Beam */}
      <mesh ref={fanRef} position={[0, -1.8, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[4.2, 3.6, 16, 1, true]} />
        <meshBasicMaterial
          color={isLight ? '#0284c7' : '#00f0ff'}
          transparent
          opacity={isLight ? 0.2 : 0.12}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}

// Expanding Acoustic Pulse Ring
function SonarPulseWave({ isLight = false }: { isLight?: boolean }) {
  const pulseRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (pulseRef.current) {
      const cycle = (state.clock.elapsedTime * 0.8) % 1;
      const radius = cycle * 14;
      pulseRef.current.scale.set(radius, radius, radius);
      const mat = pulseRef.current.material as THREE.MeshBasicMaterial;
      if (mat) {
        mat.opacity = Math.max(0, (1 - cycle) * (isLight ? 0.5 : 0.45));
      }
    }
  });

  return (
    <mesh ref={pulseRef} position={[0, 0.05, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.95, 1.0, 32]} />
      <meshBasicMaterial
        color={isLight ? '#0284c7' : '#00f0ff'}
        transparent
        opacity={0.4}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// 3D Anomaly / Detection Marker
function Anomaly3DMarker({
  detection,
  position,
  isSelected,
  onSelect,
  isLight = false,
}: {
  detection: Detection;
  position: [number, number, number];
  isSelected: boolean;
  onSelect: () => void;
  isLight?: boolean;
}) {
  const markerRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (markerRef.current) {
      markerRef.current.rotation.y = state.clock.elapsedTime * 0.6;
    }
  });

  let markerColor = isLight ? '#0284c7' : '#00f0ff';
  if (detection.class === 'ghost_gear') markerColor = '#f59e0b';
  if (detection.class === 'shipwreck') markerColor = '#ec4899';
  if (detection.class === 'unexploded_ordnance') markerColor = '#ef4444';
  if (detection.class === 'pipeline_anomaly') markerColor = '#8b5cf6';

  return (
    <group position={position} onClick={onSelect}>
      <group ref={markerRef} position={[0, 0.8, 0]}>
        <mesh>
          <octahedronGeometry args={[isSelected ? 0.45 : 0.3, 0]} />
          <meshStandardMaterial
            color={markerColor}
            emissive={markerColor}
            emissiveIntensity={isSelected ? 0.8 : 0.4}
          />
        </mesh>

        <mesh>
          <octahedronGeometry args={[isSelected ? 0.6 : 0.42, 0]} />
          <meshBasicMaterial color={markerColor} wireframe opacity={0.6} transparent />
        </mesh>
      </group>

      <mesh position={[0.4, 0.02, 0.4]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[0.8, 1.6]} />
        <meshBasicMaterial color={isLight ? '#cbd5e1' : '#020617'} opacity={0.7} transparent />
      </mesh>

      <Text
        position={[0, 1.4, 0]}
        fontSize={0.22}
        color={isSelected ? (isLight ? '#0f172a' : '#ffffff') : isLight ? '#0369a1' : '#38bdf8'}
        anchorX="center"
        anchorY="middle"
      >
        {`${detection.classNameLabel.split(' ')[0]} (${(detection.confidence * 100).toFixed(0)}%)`}
      </Text>
    </group>
  );
}

// Bathymetric Seabed Mesh
function OceanFloor({
  profile,
  isLight = false,
}: {
  profile: RenderProfile;
  isLight?: boolean;
}) {
  const segments = profile === 'HIGH' ? 64 : profile === 'BALANCED' ? 32 : 16;

  const geometry = useMemo(() => {
    const geom = new THREE.PlaneGeometry(30, 30, segments, segments);
    const pos = geom.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z =
        Math.sin(x * 0.25) * Math.cos(y * 0.25) * 0.6 +
        Math.sin(x * 0.6 + y * 0.4) * 0.2 -
        Math.sin(x * 0.1) * 0.4;
      pos.setZ(i, z);
    }
    geom.computeVertexNormals();
    return geom;
  }, [segments]);

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <mesh geometry={geometry} receiveShadow>
        <meshStandardMaterial
          color={isLight ? '#dbeafe' : '#061322'}
          roughness={0.9}
          metalness={0.1}
        />
      </mesh>

      <mesh geometry={geometry} position={[0, 0, 0.01]}>
        <meshBasicMaterial
          color={isLight ? '#0284c7' : '#0ea5e9'}
          wireframe
          opacity={isLight ? 0.25 : 0.16}
          transparent
        />
      </mesh>
    </group>
  );
}

// Marine Snow / Acoustic Particles
function MarineParticles({
  count = 200,
  isLight = false,
}: {
  count?: number;
  isLight?: boolean;
}) {
  const particles = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 25;
      pos[i * 3 + 1] = Math.random() * 5;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 25;
    }
    return pos;
  }, [count]);

  const pointsRef = useRef<THREE.Points>(null);

  useFrame((_, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.02;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[particles, 3]} count={count} array={particles} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial
        size={0.06}
        color={isLight ? '#0284c7' : '#38bdf8'}
        transparent
        opacity={isLight ? 0.6 : 0.4}
      />
    </points>
  );
}

export const ThreeOceanScene: React.FC<ThreeOceanSceneProps> = ({
  detections = [],
  selectedDetectionId,
  onSelectDetection,
  renderProfile = 'HIGH',
}) => {
  const { isLight } = useTheme();
  const particleCount = renderProfile === 'HIGH' ? 300 : renderProfile === 'BALANCED' ? 120 : 40;

  const markerPositions = useMemo(() => {
    return detections.map((det, idx) => {
      const angle = (idx / Math.max(1, detections.length)) * Math.PI * 2;
      const radius = 3.5 + (idx % 3) * 2.2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      return {
        detection: det,
        position: [x, 0, z] as [number, number, number],
      };
    });
  }, [detections]);

  const bgColor = isLight ? '#d2ecf9' : '#040913';

  return (
    <div className="relative w-full h-full min-h-[350px] bg-[#030710] dark:bg-[#030710] light:bg-[#d2ecf9] rounded-lg overflow-hidden border border-[#142338] dark:border-[#142338] light:border-sky-300 transition-colors">
      {/* 3D Scene Status Badge */}
      <div className="absolute top-3 left-3 z-10 flex items-center gap-2 bg-[#08121e]/85 dark:bg-[#08121e]/85 light:bg-white/90 backdrop-blur border border-[#172c44] dark:border-[#172c44] light:border-sky-200 px-2.5 py-1 rounded text-[11px] font-mono text-slate-300 dark:text-slate-300 light:text-slate-800 pointer-events-none shadow-sm">
        <span className="w-2 h-2 rounded-full bg-cyan-400 dark:bg-cyan-400 light:bg-sky-600 animate-ping" />
        <span>3D BATHYMETRY & SONAR SCANNER</span>
        <span className="text-cyan-400 dark:text-cyan-400 light:text-sky-700 font-bold">[{renderProfile} LOD]</span>
      </div>

      <Canvas
        camera={{ position: [0, 8, 14], fov: 45 }}
        gl={{
          antialias: renderProfile === 'HIGH',
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
        <fog attach="fog" args={[bgColor, isLight ? 10 : 8, isLight ? 32 : 28]} />

        {/* Lighting */}
        <ambientLight intensity={isLight ? 0.8 : 0.4} />
        <directionalLight
          position={[10, 20, 10]}
          intensity={isLight ? 1.0 : 0.6}
          color={isLight ? '#ffffff' : '#38bdf8'}
        />
        <spotLight
          position={[0, 10, 0]}
          angle={0.6}
          penumbra={0.8}
          intensity={isLight ? 1.5 : 1.2}
          color={isLight ? '#0284c7' : '#00f0ff'}
          castShadow
        />

        <SonarVehicle position={[0, 3.2, 0]} isLight={isLight} />
        <SonarPulseWave isLight={isLight} />
        <OceanFloor profile={renderProfile} isLight={isLight} />
        <MarineParticles count={particleCount} isLight={isLight} />

        {markerPositions.map(({ detection, position }) => (
          <Anomaly3DMarker
            key={detection.id}
            detection={detection}
            position={position}
            isSelected={detection.id === selectedDetectionId}
            onSelect={() => onSelectDetection && onSelectDetection(detection)}
            isLight={isLight}
          />
        ))}

        <OrbitControls
          maxPolarAngle={Math.PI / 2 - 0.05}
          minDistance={4}
          maxDistance={22}
          enableDamping
          dampingFactor={0.05}
        />
      </Canvas>
    </div>
  );
};
