import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, MeshDistortMaterial } from '@react-three/drei';
import * as THREE from 'three';
import { useTheme } from '../../context/ThemeContext';

function LiquidOceanCore({ isDark }: { isDark: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Group>(null);
  const outerRingRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (meshRef.current) {
      meshRef.current.rotation.y = t * 0.25;
      meshRef.current.rotation.x = Math.sin(t * 0.2) * 0.3;
    }
    if (ringRef.current) {
      ringRef.current.rotation.z = -t * 0.3;
      ringRef.current.rotation.y = t * 0.15;
    }
    if (outerRingRef.current) {
      outerRingRef.current.rotation.x = t * 0.2;
      outerRingRef.current.rotation.z = t * 0.25;
    }
  });

  return (
    <group>
      {/* 3D Liquid Dynamic Ocean Orb */}
      <Float speed={2.5} rotationIntensity={0.6} floatIntensity={1.2}>
        <mesh ref={meshRef} scale={1.8}>
          <sphereGeometry args={[1, 64, 64]} />
          <MeshDistortMaterial
            color={isDark ? '#00e5ff' : '#0284c7'}
            emissive={isDark ? '#083344' : '#bae6fd'}
            emissiveIntensity={isDark ? 0.4 : 0.2}
            roughness={0.1}
            metalness={0.8}
            distort={0.45}
            speed={3}
            transparent
            opacity={0.85}
          />
        </mesh>
      </Float>

      {/* Acoustic Wave Orbit Ring 1 */}
      <group ref={ringRef}>
        <mesh rotation={[Math.PI / 2.3, 0, 0]}>
          <torusGeometry args={[2.7, 0.03, 16, 64]} />
          <meshStandardMaterial
            color={isDark ? '#22d3ee' : '#0284c7'}
            emissive={isDark ? '#22d3ee' : '#38bdf8'}
            emissiveIntensity={0.7}
            transparent
            opacity={0.65}
          />
        </mesh>
      </group>

      {/* Acoustic Wave Orbit Ring 2 */}
      <group ref={outerRingRef}>
        <mesh rotation={[Math.PI / 3, Math.PI / 4, 0]}>
          <torusGeometry args={[3.4, 0.02, 16, 64]} />
          <meshStandardMaterial
            color={isDark ? '#34d399' : '#0d9488'}
            emissive={isDark ? '#34d399' : '#14b8a6'}
            emissiveIntensity={0.5}
            transparent
            opacity={0.5}
          />
        </mesh>
      </group>
    </group>
  );
}

export const OceanSceneHero3D: React.FC<{ className?: string }> = ({ className = '' }) => {
  const { isDark } = useTheme();

  return (
    <div className={`relative w-full h-[280px] sm:h-[340px] pointer-events-none ${className}`}>
      <Canvas
        camera={{ position: [0, 0, 7], fov: 45 }}
        gl={{ alpha: true, antialias: true }}
      >
        <ambientLight intensity={isDark ? 0.6 : 0.9} />
        <directionalLight position={[5, 8, 5]} intensity={isDark ? 1.2 : 1.5} color="#38bdf8" />
        <pointLight position={[-6, -4, -4]} intensity={1} color="#34d399" />
        <LiquidOceanCore isDark={isDark} />
      </Canvas>
    </div>
  );
};
