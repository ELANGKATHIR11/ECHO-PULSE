import React, { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import { Detection, Mission } from '../../types';
import { formatDMS } from '../../utils/sonarProcessor';
import { useTheme } from '../../context/ThemeContext';

export interface DigitalTwinLayers {
  bathymetry: boolean;
  sonarBeam: boolean;
  sonarPulse: boolean;
  detections: boolean;
  shadows: boolean;
  heatmap: boolean;
  contours: boolean;
  grid: boolean;
  vessel: boolean;
  particles: boolean;
}

export type DigitalTwinCameraMode = 'FREE_ORBIT' | 'FOLLOW_AUV' | 'FOLLOW_VESSEL' | 'PLAN_VIEW' | 'SIDE_PROFILE';
export type DigitalTwinColorScheme = 'OCEANIC' | 'BATHYMETRIC_DEM' | 'THERMAL_SNR' | 'ABYSS';
export type SonarPulseMode = 'DUAL_COMBINED' | 'SWATH_SWEEP' | 'CHIRP_RADIAL' | 'SECTOR_RADAR';

export interface SonarShaderConfig {
  pulseMode: SonarPulseMode;
  pulseSpeed: number; // 0.5 to 3.0
  pulseFrequency: number; // 1.5 to 4.5 (corresponds to kHz)
  pulseIntensity: number; // 0.5 to 2.5
  swathWidth: number;
  lastPingTimestamp: number;
}

interface DigitalTwinCanvasProps {
  mission: Mission;
  allMissions?: Mission[];
  detections?: Detection[];
  selectedDetectionId?: string | null;
  onSelectDetection?: (detection: Detection) => void;
  layers: DigitalTwinLayers;
  cameraMode: DigitalTwinCameraMode;
  colorScheme: DigitalTwinColorScheme;
  playbackProgress?: number; // 0 to 1 for trajectory timeline
  onHoverPoint?: (info: { depthMeters: number; lat: number; lng: number } | null) => void;
  sonarConfig?: SonarShaderConfig;
  onAuvPositionChange?: (pos: { x: number; y: number; z: number }) => void;
}

// GLSL Vertex Shader for Procedural Seabed & Sonar Mapping
const seabedVertexShader = `
  varying vec3 vWorldPosition;
  varying vec3 vNormal;
  varying vec2 vUv;
  varying float vElevation;

  void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPos.xyz;
    vElevation = position.z;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

// GLSL Fragment Shader for Multi-layer Bathymetry, Acoustic Beam Sweeps & CHIRP Pulses
const seabedFragmentShader = `
  uniform float uTime;
  uniform vec3 uAuvPos;
  uniform float uSonarPulseEnabled;
  uniform int uPulseMode; // 0: CHIRP_RADIAL, 1: SWATH_SWEEP, 2: SECTOR_RADAR, 3: DUAL_COMBINED
  uniform float uPulseSpeed;
  uniform float uPulseFrequency;
  uniform float uPulseIntensity;
  uniform float uSwathWidth;
  uniform float uManualPingTime;
  uniform int uColorScheme; // 0: Oceanic, 1: GEBCO, 2: Thermal, 3: Abyss
  uniform float uContoursEnabled;
  uniform float uGridEnabled;
  uniform float uHeatmapEnabled;

  varying vec3 vWorldPosition;
  varying vec3 vNormal;
  varying vec2 vUv;
  varying float vElevation;

  void main() {
    // 1. Base Elevation Colormap
    float normZ = clamp((vElevation + 2.5) / 6.0, 0.0, 1.0);
    vec3 baseColor = vec3(0.02, 0.12, 0.28);

    if (uColorScheme == 1) {
      // GEBCO Bathymetric DEM Gradient
      if (normZ < 0.2) {
        baseColor = vec3(0.04, 0.12, 0.65);
      } else if (normZ < 0.4) {
        baseColor = vec3(0.02, 0.52, 0.82);
      } else if (normZ < 0.6) {
        baseColor = vec3(0.08, 0.78, 0.42);
      } else if (normZ < 0.8) {
        baseColor = vec3(0.92, 0.76, 0.12);
      } else {
        baseColor = vec3(0.92, 0.22, 0.18);
      }
    } else if (uColorScheme == 2) {
      // Thermal Acoustic SNR
      baseColor = vec3(
        min(1.0, normZ * 1.35),
        min(1.0, (1.0 - normZ) * 0.65 + 0.08),
        0.22
      );
    } else if (uColorScheme == 3) {
      // Deep Abyss Midnight
      baseColor = vec3(
        0.01,
        0.03 + normZ * 0.12,
        0.10 + normZ * 0.32
      );
    } else {
      // Oceanic Cyan Default
      baseColor = vec3(
        0.02 + normZ * 0.08,
        0.11 + normZ * 0.52,
        0.26 + normZ * 0.56
      );
    }

    // Directional lighting & acoustic grazing angle calculations
    vec3 lightDir = normalize(vec3(0.35, 0.85, 0.45));
    float diff = max(dot(vNormal, lightDir), 0.22);
    
    // Dynamic Subsea Underwater Caustic Ripples from surface waves
    float caustic1 = sin(vWorldPosition.x * 1.4 + uTime * 0.9) * sin(vWorldPosition.z * 1.4 + uTime * 0.7);
    float caustic2 = sin(vWorldPosition.x * 2.8 - uTime * 1.2) * cos(vWorldPosition.z * 2.8 + uTime * 1.1);
    float caustics = clamp((caustic1 + caustic2) * 0.5 + 0.5, 0.0, 1.0);
    caustics = pow(caustics, 3.0) * 0.28;
    
    // Water Depth Color Absorption (Deep Rayleigh scattering)
    vec3 waterFogColor = vec3(0.01, 0.06, 0.16);
    float depthAbsorption = clamp(abs(vWorldPosition.y) / 10.0, 0.0, 0.85);
    vec3 illuminatedColor = (baseColor + vec3(0.12, 0.75, 0.95) * caustics) * (diff * 0.75 + 0.25);
    vec3 finalColor = mix(illuminatedColor, waterFogColor, depthAbsorption * 0.35);

    // 2. Depth Contour Isolines (Anti-aliased Analytical Derivatives)
    if (uContoursEnabled > 0.5) {
      float contourScale = 2.4;
      float line = abs(fract(vElevation * contourScale - 0.5) - 0.5) / max(fwidth(vElevation * contourScale), 0.0001);
      float contour = 1.0 - clamp(line, 0.0, 1.0);
      finalColor += vec3(0.22, 0.75, 1.0) * contour * 0.38;
    }

    // 3. Spatial Coordinate Grid Overlay
    if (uGridEnabled > 0.5) {
      vec2 gridUv = abs(fract(vWorldPosition.xz * 0.5 - 0.5) - 0.5) / max(fwidth(vWorldPosition.xz * 0.5), vec2(0.0001));
      float gridLine = 1.0 - clamp(min(gridUv.x, gridUv.y), 0.0, 1.0);
      finalColor += vec3(0.0, 0.55, 0.85) * gridLine * 0.22;
    }

    // 4. SHADER-BASED SONAR PULSE & ACOUSTIC SCANNING BEAM ENGINE
    if (uSonarPulseEnabled > 0.5) {
      vec2 auvGround = uAuvPos.xz;
      vec2 targetGround = vWorldPosition.xz;
      float dist = length(targetGround - auvGround);

      vec3 pingCyan = vec3(0.0, 0.95, 1.0);
      vec3 swathAmber = vec3(0.2, 0.9, 1.0);
      vec3 highEnergyYellow = vec3(1.0, 0.88, 0.25);

      // Acoustic reflection grazing highlight (facing AUV)
      vec3 toAuv = normalize(uAuvPos - vWorldPosition);
      float acousticEchoFactor = max(0.0, dot(vNormal, toAuv));

      // A. CONCENTRIC CHIRP PULSE WAVEFRONTS (Modes 0 and 3)
      if (uPulseMode == 0 || uPulseMode == 3) {
        float speed = uPulseSpeed * 5.0;
        float wavePhase = dist * uPulseFrequency - uTime * speed;
        float wave = sin(wavePhase);

        // Leading shockwave ring with exponential tail
        float pulseLeading = smoothstep(0.35, 0.98, wave);
        float pulseDistanceDecay = exp(-dist * 0.075);

        // High frequency acoustic micro-texture reverberation
        float microRipples = sin(dist * uPulseFrequency * 3.5 - uTime * speed * 2.2) * 0.5 + 0.5;
        float waveEnergy = (pulseLeading * 0.85 + microRipples * 0.35 * pulseLeading) * pulseDistanceDecay * uPulseIntensity;

        // Add acoustic wave illuminating seabed features
        finalColor += (pingCyan * 1.3 + vec3(0.4, 1.0, 0.7) * acousticEchoFactor * 0.7) * waveEnergy;
      }

      // B. SIDE-SCAN SONAR SWATH FAN BEAM (Modes 1 and 3)
      if (uPulseMode == 1 || uPulseMode == 3) {
        // Focused lateral along-track beam along X axis
        float alongTrackDist = abs(targetGround.x - auvGround.x);
        float beamWidth = 0.52;
        float beamCore = exp(-pow(alongTrackDist / beamWidth, 2.0));

        // Lateral swath boundary (port & starboard limits)
        float lateralDist = abs(targetGround.y - auvGround.y);
        float swathCoverage = smoothstep(uSwathWidth * 0.65, 0.0, lateralDist);

        // Dynamic sonar ping ripple texture along the swath line
        float swathScanRipples = sin(targetGround.y * 7.0 - uTime * 18.0) * 0.2 + 0.8;

        // Ultra-sharp laser footprint center line
        float laserCenter = smoothstep(0.09, 0.0, alongTrackDist);

        vec3 swathLighting = swathAmber * (beamCore * swathScanRipples + laserCenter * 1.8) * swathCoverage * uPulseIntensity * 1.3;
        finalColor += swathLighting;
      }

      // C. 360° ROTATIONAL SECTOR RADAR/HYDRO-SONAR BEAM (Mode 2)
      if (uPulseMode == 2) {
        vec2 dir = targetGround - auvGround;
        float angle = atan(dir.y, dir.x);
        float sweepAngle = mod(uTime * uPulseSpeed * 1.6, 6.283185) - 3.141592;
        float angleDiff = mod(angle - sweepAngle + 6.283185, 6.283185);

        // Phosphor persistence decay trail
        float radarTrail = exp(-angleDiff * 2.4);
        float radarLead = smoothstep(0.07, 0.0, abs(angleDiff));
        float rangeGate = smoothstep(24.0, 1.0, dist);

        vec3 radarLight = pingCyan * (radarTrail * 0.9 + radarLead * 2.2) * rangeGate * uPulseIntensity;
        finalColor += radarLight;
      }

      // D. MANUAL HIGH-ENERGY SHOCKWAVE TRIGGER
      if (uManualPingTime > 0.0) {
        float pingAge = uTime - uManualPingTime;
        if (pingAge >= 0.0 && pingAge < 3.2) {
          float shockRadius = pingAge * 10.5;
          float ringDist = abs(dist - shockRadius);
          float ringSharpness = smoothstep(0.8, 0.0, ringDist);
          float ringFade = 1.0 - (pingAge / 3.2);

          vec3 shockColor = highEnergyYellow * ringSharpness * ringFade * 3.2 * uPulseIntensity;
          finalColor += shockColor;
        }
      }
    }

    // 5. Ping Density Heatmap Overlay
    if (uHeatmapEnabled > 0.5 && normZ > 0.65) {
      finalColor = mix(finalColor, vec3(0.95, 0.2, 0.4), 0.52);
    }

    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

// 3D Survey Vessel on Water Surface
function SurfaceVessel({ progress = 0.5, enabled = true }: { progress?: number; enabled?: boolean }) {
  const vesselRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (vesselRef.current && enabled) {
      // Gentle surface wave pitching & rolling
      vesselRef.current.position.y = 8.5 + Math.sin(state.clock.elapsedTime * 1.2) * 0.12;
      vesselRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.9) * 0.03;
      vesselRef.current.rotation.x = Math.cos(state.clock.elapsedTime * 0.7) * 0.02;

      // Trajectory along X axis
      vesselRef.current.position.x = -8 + progress * 16;
    }
  });

  if (!enabled) return null;

  return (
    <group ref={vesselRef} position={[-8, 8.5, -2]}>
      {/* Ship Hull */}
      <mesh castShadow position={[0, 0, 0]}>
        <boxGeometry args={[3.6, 0.9, 1.3]} />
        <meshStandardMaterial color="#0f172a" metalness={0.8} roughness={0.3} />
      </mesh>
      {/* Bow slope */}
      <mesh position={[2.1, 0.1, 0]} rotation={[0, 0, -0.4]}>
        <boxGeometry args={[1.2, 0.8, 1.2]} />
        <meshStandardMaterial color="#0284c7" metalness={0.7} />
      </mesh>
      {/* Bridge Superstructure */}
      <mesh position={[-0.4, 0.8, 0]}>
        <boxGeometry args={[1.4, 0.9, 1.0]} />
        <meshStandardMaterial color="#e2e8f0" metalness={0.5} roughness={0.4} />
      </mesh>
      {/* Radar Mast */}
      <mesh position={[-0.3, 1.5, 0]}>
        <cylinderGeometry args={[0.04, 0.04, 0.8]} />
        <meshBasicMaterial color="#38bdf8" />
      </mesh>
      {/* Tow Cable descending into deep water */}
      <mesh position={[-1.6, -4.2, 0]}>
        <cylinderGeometry args={[0.02, 0.02, 8.5]} />
        <meshBasicMaterial color="#00f0ff" opacity={0.6} transparent />
      </mesh>
    </group>
  );
}

// Subsea AUV / Towfish with Spinning Propeller & Sonar Transducers
function SubseaAUV({
  progress = 0.5,
  showSonarBeam = true,
  onPositionUpdate,
}: {
  progress?: number;
  showSonarBeam?: boolean;
  onPositionUpdate?: (pos: THREE.Vector3) => void;
}) {
  const auvRef = useRef<THREE.Group>(null);
  const propRef = useRef<THREE.Mesh>(null);
  const fanPortRef = useRef<THREE.Mesh>(null);
  const fanStarboardRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (auvRef.current) {
      // Dynamic subsea trajectory
      const currentX = -8 + progress * 16;
      const currentZ = Math.sin(progress * Math.PI * 2) * 1.5;
      const currentY = 3.2 + Math.sin(state.clock.elapsedTime * 0.8) * 0.15;

      auvRef.current.position.set(currentX, currentY, currentZ);
      auvRef.current.rotation.y = Math.PI / 2 + Math.cos(progress * Math.PI * 2) * 0.2;
      auvRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.5) * 0.03;

      if (onPositionUpdate) {
        onPositionUpdate(auvRef.current.position);
      }
    }

    if (propRef.current) {
      propRef.current.rotation.x += 0.35;
    }

    if (fanPortRef.current && fanStarboardRef.current) {
      const pulse = (Math.sin(state.clock.elapsedTime * 4) + 1) * 0.5;
      fanPortRef.current.scale.set(1 + pulse * 0.15, 1, 1 + pulse * 0.15);
      fanStarboardRef.current.scale.set(1 + pulse * 0.15, 1, 1 + pulse * 0.15);
    }
  });

  return (
    <group ref={auvRef} position={[-8, 3.2, 0]}>
      {/* Torpedo Titanium Body */}
      <mesh castShadow rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.26, 0.28, 2.4, 20]} />
        <meshStandardMaterial color="#0284c7" metalness={0.9} roughness={0.15} />
      </mesh>

      {/* High-visibility Fluorescent Nose Cone */}
      <mesh position={[1.35, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.26, 0.5, 20]} />
        <meshStandardMaterial color="#22d3ee" emissive="#06b6d4" emissiveIntensity={0.4} metalness={0.9} />
      </mesh>

      {/* Side-Scan Sonar Port & Starboard Transducers */}
      <mesh position={[0, 0, 0.32]}>
        <boxGeometry args={[1.4, 0.12, 0.12]} />
        <meshStandardMaterial color="#f59e0b" metalness={0.95} roughness={0.1} />
      </mesh>
      <mesh position={[0, 0, -0.32]}>
        <boxGeometry args={[1.4, 0.12, 0.12]} />
        <meshStandardMaterial color="#f59e0b" metalness={0.95} roughness={0.1} />
      </mesh>

      {/* Stabilizer Fins */}
      <mesh position={[-1.0, 0, 0]}>
        <boxGeometry args={[0.4, 0.9, 0.05]} />
        <meshStandardMaterial color="#0f172a" />
      </mesh>
      <mesh position={[-1.0, 0, 0]}>
        <boxGeometry args={[0.4, 0.05, 0.9]} />
        <meshStandardMaterial color="#0f172a" />
      </mesh>

      {/* Thruster Propeller */}
      <mesh ref={propRef} position={[-1.3, 0, 0]}>
        <boxGeometry args={[0.06, 0.45, 0.1]} />
        <meshStandardMaterial color="#f8fafc" metalness={0.9} />
      </mesh>

      {/* AUV Navigation Beacon Light */}
      <pointLight position={[0, 0.4, 0]} color="#00f0ff" intensity={1.8} distance={8} />

      {/* Volumetric Dual Side-Scan Acoustic Fan Beams */}
      {showSonarBeam && (
        <group position={[0, -0.1, 0]}>
          {/* Port Acoustic Swath Cone */}
          <mesh
            ref={fanPortRef}
            position={[0, -2.4, 3.2]}
            rotation={[Math.PI / 2.8, 0, 0]}
          >
            <coneGeometry args={[3.2, 5.2, 24, 1, true]} />
            <meshBasicMaterial
              color="#00f0ff"
              transparent
              opacity={0.14}
              side={THREE.DoubleSide}
              wireframe={false}
            />
          </mesh>

          {/* Starboard Acoustic Swath Cone */}
          <mesh
            ref={fanStarboardRef}
            position={[0, -2.4, -3.2]}
            rotation={[-Math.PI / 2.8, 0, 0]}
          >
            <coneGeometry args={[3.2, 5.2, 24, 1, true]} />
            <meshBasicMaterial
              color="#00f0ff"
              transparent
              opacity={0.14}
              side={THREE.DoubleSide}
              wireframe={false}
            />
          </mesh>

          {/* Ground Footprint Swath Laser Line */}
          <mesh position={[0, -3.15, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[0.3, 11.5]} />
            <meshBasicMaterial color="#22d3ee" transparent opacity={0.65} side={THREE.DoubleSide} />
          </mesh>
        </group>
      )}
    </group>
  );
}

// Shader-driven Procedural Seabed Bathymetric Terrain Engine with Dynamic Acoustic Pulse
function ProceduralSeabedTerrain({
  colorScheme,
  contours,
  grid,
  heatmap,
  sonarPulseEnabled,
  sonarConfig,
  auvPosition,
  onPointerMove,
}: {
  colorScheme: DigitalTwinColorScheme;
  contours: boolean;
  grid: boolean;
  heatmap: boolean;
  sonarPulseEnabled: boolean;
  sonarConfig?: SonarShaderConfig;
  auvPosition: THREE.Vector3;
  onPointerMove?: (e: any) => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);

  // Generate multi-scale bathymetric geometry with procedural trenches & ridges
  const { geometry } = useMemo(() => {
    const size = 38;
    const segments = 96;
    const geom = new THREE.PlaneGeometry(size, size, segments, segments);
    const pos = geom.attributes.position;

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);

      // Multi-scale procedural ocean bathymetry (trenches, ridges, seamounts)
      const trench = Math.sin(x * 0.12) * Math.cos(y * 0.12) * 1.8;
      const seaRidge = Math.sin(x * 0.35 + y * 0.25) * 0.85;
      const ripple = Math.sin(x * 0.8) * Math.sin(y * 0.8) * 0.25;
      const mound = Math.exp(-((x - 3) ** 2 + (y + 4) ** 2) * 0.04) * 2.2;
      const z = trench + seaRidge + ripple + mound - 0.4;

      pos.setZ(i, z);
    }

    geom.computeVertexNormals();
    return { geometry: geom };
  }, []);

  // Map color scheme & pulse mode to integer enum for GLSL uniform
  const colorSchemeInt = useMemo(() => {
    switch (colorScheme) {
      case 'BATHYMETRIC_DEM':
        return 1;
      case 'THERMAL_SNR':
        return 2;
      case 'ABYSS':
        return 3;
      default:
        return 0; // OCEANIC
    }
  }, [colorScheme]);

  const pulseModeInt = useMemo(() => {
    if (!sonarConfig) return 3; // DUAL_COMBINED
    switch (sonarConfig.pulseMode) {
      case 'CHIRP_RADIAL':
        return 0;
      case 'SWATH_SWEEP':
        return 1;
      case 'SECTOR_RADAR':
        return 2;
      case 'DUAL_COMBINED':
      default:
        return 3;
    }
  }, [sonarConfig?.pulseMode]);

  // Persistent uniforms object
  const uniforms = useMemo(() => {
    return {
      uTime: { value: 0 },
      uAuvPos: { value: new THREE.Vector3(0, 3.2, 0) },
      uSonarPulseEnabled: { value: 1.0 },
      uPulseMode: { value: 3 },
      uPulseSpeed: { value: 1.2 },
      uPulseFrequency: { value: 2.2 },
      uPulseIntensity: { value: 1.4 },
      uSwathWidth: { value: 24.0 },
      uManualPingTime: { value: -100.0 },
      uColorScheme: { value: 0 },
      uContoursEnabled: { value: 1.0 },
      uGridEnabled: { value: 1.0 },
      uHeatmapEnabled: { value: 0.0 },
    };
  }, []);

  // Keep uniforms in sync on each render
  useEffect(() => {
    uniforms.uSonarPulseEnabled.value = sonarPulseEnabled ? 1.0 : 0.0;
    uniforms.uPulseMode.value = pulseModeInt;
    uniforms.uPulseSpeed.value = sonarConfig?.pulseSpeed ?? 1.2;
    uniforms.uPulseFrequency.value = sonarConfig?.pulseFrequency ?? 2.2;
    uniforms.uPulseIntensity.value = sonarConfig?.pulseIntensity ?? 1.4;
    uniforms.uSwathWidth.value = sonarConfig?.swathWidth ?? 24.0;
    uniforms.uColorScheme.value = colorSchemeInt;
    uniforms.uContoursEnabled.value = contours ? 1.0 : 0.0;
    uniforms.uGridEnabled.value = grid ? 1.0 : 0.0;
    uniforms.uHeatmapEnabled.value = heatmap ? 1.0 : 0.0;
    if (sonarConfig?.lastPingTimestamp) {
      // Pass normalized elapsed time for the ping
      uniforms.uManualPingTime.value = uniforms.uTime.value;
    }
  }, [
    sonarPulseEnabled,
    pulseModeInt,
    sonarConfig?.pulseSpeed,
    sonarConfig?.pulseFrequency,
    sonarConfig?.pulseIntensity,
    sonarConfig?.swathWidth,
    sonarConfig?.lastPingTimestamp,
    colorSchemeInt,
    contours,
    grid,
    heatmap,
    uniforms,
  ]);

  // Frame tick to animate shader time and dynamic AUV tracking
  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
      materialRef.current.uniforms.uAuvPos.value.copy(auvPosition);
    }
  });

  return (
    <group rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
      {/* Main Shader-powered Seabed Mesh */}
      <mesh
        ref={meshRef}
        geometry={geometry}
        receiveShadow
        onPointerMove={onPointerMove}
      >
        <shaderMaterial
          ref={materialRef}
          vertexShader={seabedVertexShader}
          fragmentShader={seabedFragmentShader}
          uniforms={uniforms}
          wireframe={false}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}

// 3D Seabed Target Marker with Acoustic Shadow and Laser Pin
function SubseaAnomalyMarker({
  detection,
  position,
  isSelected,
  showShadow,
  onSelect,
}: {
  detection: Detection;
  position: [number, number, number];
  isSelected: boolean;
  showShadow: boolean;
  onSelect: () => void;
}) {
  const markerRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (markerRef.current) {
      markerRef.current.rotation.y = state.clock.elapsedTime * 0.7;
    }
  });

  let color = '#22d3ee';
  let badgeLabel = 'DET';
  if (detection.class === 'ghost_gear') {
    color = '#f59e0b';
    badgeLabel = 'NET';
  } else if (detection.class === 'shipwreck') {
    color = '#ec4899';
    badgeLabel = 'WRECK';
  } else if (detection.class === 'unexploded_ordnance') {
    color = '#ef4444';
    badgeLabel = 'UXO';
  } else if (detection.class === 'pipeline_anomaly') {
    color = '#8b5cf6';
    badgeLabel = 'PIPE';
  }

  return (
    <group position={position}>
      {/* Laser Targeting Guide Vector to Seabed */}
      <mesh position={[0, 1.0, 0]}>
        <cylinderGeometry args={[0.02, 0.02, 2.0]} />
        <meshBasicMaterial color={color} transparent opacity={isSelected ? 0.9 : 0.4} />
      </mesh>

      {/* Rotating 3D Diamond Beacon */}
      <group ref={markerRef} position={[0, 2.1, 0]} onClick={onSelect}>
        <mesh>
          <octahedronGeometry args={[isSelected ? 0.55 : 0.38, 0]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={isSelected ? 1.0 : 0.5}
            metalness={0.8}
            roughness={0.2}
          />
        </mesh>
        <mesh>
          <octahedronGeometry args={[isSelected ? 0.72 : 0.52, 0]} />
          <meshBasicMaterial color={color} wireframe opacity={0.6} transparent />
        </mesh>
      </group>

      {/* Acoustic Shadow Ground Footprint */}
      {showShadow && (
        <mesh position={[0.8, 0.04, 0.8]} rotation={[-Math.PI / 2, 0, 0.5]}>
          <planeGeometry args={[1.6, 3.4]} />
          <meshBasicMaterial color="#01040a" opacity={0.9} transparent />
        </mesh>
      )}

      {/* Floating Target Label HUD */}
      <Html position={[0, 3.0, 0]} center distanceFactor={16}>
        <div
          onClick={onSelect}
          className={`px-2 py-0.5 rounded font-mono text-[10px] font-bold uppercase tracking-wider flex items-center gap-1.5 shadow-2xl cursor-pointer transition-all ${
            isSelected
              ? 'bg-[#040D1B] text-white border-2 border-cyan-400 scale-110 shadow-[0_0_16px_rgba(34,211,238,0.6)]'
              : 'bg-[#020712]/90 text-cyan-300 border border-cyan-500/40 hover:border-cyan-400'
          }`}
        >
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: color }}
          />
          <span>{badgeLabel}</span>
          <span className="text-slate-400">{(detection.confidence * 100).toFixed(0)}%</span>
        </div>
      </Html>
    </group>
  );
}

// Suspended Marine Snow Particle Cloud with organic Brownian drift & turbulence
function SuspendedMarineSnow({ count = 360, enabled = true }: { count?: number; enabled?: boolean }) {
  const particles = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 36;
      pos[i * 3 + 1] = Math.random() * 9.5;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 36;
    }
    return pos;
  }, [count]);

  const pointsRef = useRef<THREE.Points>(null);

  useFrame((state, delta) => {
    if (pointsRef.current && enabled) {
      // Organic ocean current drift
      pointsRef.current.rotation.y += delta * 0.012;
      const positions = pointsRef.current.geometry.attributes.position.array as Float32Array;
      const time = state.clock.elapsedTime;
      for (let i = 0; i < count; i++) {
        const yIdx = i * 3 + 1;
        positions[yIdx] -= delta * 0.15; // Slow settling drift
        if (positions[yIdx] < 0.1) {
          positions[yIdx] = 9.0; // Wrap back to surface
        }
        // Micro horizontal turbulence
        positions[i * 3] += Math.sin(time * 0.5 + i) * 0.003;
      }
      pointsRef.current.geometry.attributes.position.needsUpdate = true;
    }
  });

  if (!enabled) return null;

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[particles, 3]}
          count={count}
          array={particles}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.08} color="#7dd3fc" transparent opacity={0.45} sizeAttenuation />
    </points>
  );
}

// Realistic Animated Ocean Surface Mesh (Viewed from underneath)
function RealisticOceanWaterSurface() {
  const waterRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (waterRef.current) {
      waterRef.current.position.y = 8.8 + Math.sin(state.clock.elapsedTime * 0.8) * 0.06;
    }
  });

  return (
    <mesh ref={waterRef} position={[0, 8.8, 0]} rotation={[Math.PI / 2, 0, 0]}>
      <planeGeometry args={[44, 44, 32, 32]} />
      <meshStandardMaterial
        color="#0369a1"
        roughness={0.1}
        metalness={0.85}
        transparent
        opacity={0.35}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// Volumetric Sunbeams / Caustic Light Rays filtering down from surface
function VolumetricCausticLightRays() {
  const raysRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (raysRef.current) {
      raysRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.2) * 0.08;
    }
  });

  return (
    <group ref={raysRef} position={[0, 9, 0]}>
      {[-8, -4, 0, 4, 8].map((x, i) => (
        <mesh key={i} position={[x, -4.5, (i % 2 === 0 ? 2 : -2)]} rotation={[0.08, 0, -0.04 * x]}>
          <cylinderGeometry args={[0.3, 3.2, 9.5, 16, 1, true]} />
          <meshBasicMaterial
            color="#38bdf8"
            transparent
            opacity={0.05}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      ))}
    </group>
  );
}

// Camera Director Component to handle cinematic tracking modes
function CameraDirector({
  mode,
  auvPosition,
}: {
  mode: DigitalTwinCameraMode;
  auvPosition: THREE.Vector3;
}) {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);

  useFrame(() => {
    if (mode === 'FOLLOW_AUV') {
      camera.position.lerp(
        new THREE.Vector3(auvPosition.x - 5, auvPosition.y + 2.5, auvPosition.z + 4),
        0.05
      );
      if (controlsRef.current) {
        controlsRef.current.target.lerp(auvPosition, 0.05);
        controlsRef.current.update();
      }
    } else if (mode === 'PLAN_VIEW') {
      camera.position.lerp(new THREE.Vector3(0, 24, 0.01), 0.05);
      if (controlsRef.current) {
        controlsRef.current.target.lerp(new THREE.Vector3(0, 0, 0), 0.05);
        controlsRef.current.update();
      }
    } else if (mode === 'SIDE_PROFILE') {
      camera.position.lerp(new THREE.Vector3(0, 4, 20), 0.05);
      if (controlsRef.current) {
        controlsRef.current.target.lerp(new THREE.Vector3(0, 3, 0), 0.05);
        controlsRef.current.update();
      }
    }
  });

  return (
    <OrbitControls
      ref={controlsRef}
      maxPolarAngle={mode === 'PLAN_VIEW' ? 0.01 : Math.PI / 2 - 0.05}
      minDistance={3}
      maxDistance={35}
      enableDamping
      dampingFactor={0.06}
    />
  );
}

export const DigitalTwinCanvas: React.FC<DigitalTwinCanvasProps> = ({
  mission,
  allMissions = [],
  detections = [],
  selectedDetectionId,
  onSelectDetection,
  layers,
  cameraMode,
  colorScheme,
  playbackProgress = 0.5,
  onHoverPoint,
  sonarConfig,
  onAuvPositionChange,
}) => {
  const { isLight } = useTheme();
  const [auvPos, setAuvPos] = useState<THREE.Vector3>(new THREE.Vector3(0, 3.2, 0));

  const handleAuvPosUpdate = (pos: THREE.Vector3) => {
    setAuvPos(pos.clone());
    if (onAuvPositionChange) {
      onAuvPositionChange({ x: pos.x, y: pos.y, z: pos.z });
    }
  };

  // Compute 3D target coordinates across the terrain with exact multi-object optical spatial positioning
  const target3DNodes = useMemo(() => {
    return detections.map((det, idx) => {
      // If detection carries exact optical 3D coordinates from webcam sensor fusion
      const customWorld3D = (det as any).world3D as [number, number, number] | undefined;
      if (customWorld3D && customWorld3D.length === 3) {
        return {
          detection: det,
          position: [customWorld3D[0], 0.2, customWorld3D[2]] as [number, number, number],
        };
      }

      // If latitude and longitude offsets are available relative to mission center
      if (det.latitude && det.longitude && mission.coordinates) {
        const deltaLat = det.latitude - mission.coordinates[0];
        const deltaLng = det.longitude - mission.coordinates[1];
        const scale = 5000.0;
        const x = deltaLng * scale;
        const z = deltaLat * scale;
        const boundedX = Math.max(-16, Math.min(16, x));
        const boundedZ = Math.max(-16, Math.min(16, z));
        return {
          detection: det,
          position: [boundedX, 0.2, boundedZ] as [number, number, number],
        };
      }

      // Default radial dispersion for multiple simultaneous targets
      const angle = (idx / Math.max(1, detections.length)) * Math.PI * 2 + (idx * 0.4);
      const radius = 3.5 + (idx % 5) * 2.2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = 0.2; // ground contact

      return {
        detection: det,
        position: [x, y, z] as [number, number, number],
      };
    });
  }, [detections, mission.coordinates]);


  const bgColor = isLight ? '#d2ecf9' : '#020712';

  return (
    <div className="relative w-full h-full min-h-[500px] bg-[#020712] dark:bg-[#020712] light:bg-[#d2ecf9] rounded-xl overflow-hidden border border-cyan-500/20 dark:border-cyan-500/20 light:border-sky-300 shadow-[0_12px_48px_rgba(0,0,0,0.8)] dark:shadow-[0_12px_48px_rgba(0,0,0,0.8)] light:shadow-lg transition-colors">
      <Canvas
        camera={{ position: [0, 11, 19], fov: 45 }}
        dpr={[1, 2]}
        gl={{
          antialias: true,
          powerPreference: 'high-performance',
          preserveDrawingBuffer: false,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: isLight ? 1.05 : 1.25,
        }}
        onCreated={({ gl }) => {
          gl.domElement.addEventListener('webglcontextlost', (e) => {
            e.preventDefault();
          });
        }}
      >
        <color attach="background" args={[bgColor]} />
        <fog attach="fog" args={[bgColor, isLight ? 12 : 10, isLight ? 42 : 38]} />

        {/* Dynamic Ocean Lighting */}
        <ambientLight intensity={isLight ? 0.75 : 0.45} />
        <directionalLight
          position={[10, 25, 10]}
          intensity={isLight ? 1.0 : 0.7}
          color={isLight ? '#ffffff' : '#38bdf8'}
        />
        <spotLight
          position={[0, 14, 0]}
          angle={0.75}
          penumbra={0.9}
          intensity={isLight ? 1.8 : 1.4}
          color={isLight ? '#0284c7' : '#00f0ff'}
        />

        {/* Volumetric Sunlight Rays & Ocean Surface Mesh */}
        <RealisticOceanWaterSurface />
        <VolumetricCausticLightRays />

        {/* Surface Survey Vessel */}
        <SurfaceVessel progress={playbackProgress} enabled={layers.vessel} />

        {/* Subsea AUV / Towfish */}
        <SubseaAUV
          progress={playbackProgress}
          showSonarBeam={layers.sonarBeam}
          onPositionUpdate={handleAuvPosUpdate}
        />

        {/* Procedural Seabed Bathymetric Terrain with Shader-based Sonar Pulse */}
        {layers.bathymetry && (
          <ProceduralSeabedTerrain
            colorScheme={colorScheme}
            contours={layers.contours}
            grid={layers.grid}
            heatmap={layers.heatmap}
            sonarPulseEnabled={layers.sonarPulse}
            sonarConfig={sonarConfig}
            auvPosition={auvPos}
            onPointerMove={(e) => {
              if (onHoverPoint && e.point) {
                onHoverPoint({
                  depthMeters: Number((42.5 - e.point.y * 2.8).toFixed(1)),
                  lat: mission.coordinates[0] + e.point.x * 0.0002,
                  lng: mission.coordinates[1] + e.point.z * 0.0002,
                });
              }
            }}
          />
        )}

        {/* Subsea Target Anomalies */}
        {layers.detections &&
          target3DNodes.map(({ detection, position }) => (
            <SubseaAnomalyMarker
              key={detection.id}
              detection={detection}
              position={position}
              isSelected={detection.id === selectedDetectionId}
              showShadow={layers.shadows}
              onSelect={() => onSelectDetection && onSelectDetection(detection)}
            />
          ))}

        {/* Suspended Marine Snow */}
        <SuspendedMarineSnow enabled={layers.particles} />

        {/* Camera Tracking Director */}
        <CameraDirector mode={cameraMode} auvPosition={auvPos} />
      </Canvas>
    </div>
  );
};
