import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Shield,
  Compass,
  Radio,
  Anchor,
  Sparkles,
  Waves,
  Zap,
  Globe2,
  Fish,
  Building2,
  Award,
  ArrowRight,
  Database,
  Cpu,
  Eye,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Ship,
  MapPin,
  TrendingUp,
  Boxes
} from 'lucide-react';
import { GlassCard, GlassBadge, GlassButton, GlassPanel } from '../components/glass/GlassCard';

export const HomePage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="flex-1 min-h-screen bg-[#020611] dark:bg-[#020611] light:bg-[#f0f7fc] text-slate-100 dark:text-slate-100 light:text-slate-800 font-sans pb-20 selection:bg-cyan-500/30 selection:text-cyan-200 relative overflow-hidden transition-colors duration-300">
      {/* Background Ambient Glow & Waves */}
      <div className="absolute top-0 left-1/4 w-[700px] h-[500px] bg-radial from-cyan-500/15 dark:from-cyan-500/15 light:from-sky-400/20 via-sky-600/5 to-transparent blur-3xl pointer-events-none" />
      <div className="absolute top-1/3 right-10 w-[600px] h-[600px] bg-radial from-emerald-500/10 dark:from-emerald-500/10 light:from-teal-400/20 via-teal-600/5 to-transparent blur-3xl pointer-events-none" />
      <div className="absolute bottom-10 left-10 w-[800px] h-[500px] bg-radial from-blue-600/10 dark:from-blue-600/10 light:from-cyan-300/20 to-transparent blur-3xl pointer-events-none" />

      {/* HERO SECTION */}
      <section className="relative pt-12 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="flex flex-col items-center text-center space-y-6">
          {/* National Initiative Badge */}
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-950/60 dark:bg-cyan-950/60 light:bg-sky-100 border border-cyan-400/40 dark:border-cyan-400/40 light:border-sky-300 backdrop-blur-xl shadow-[0_0_20px_rgba(34,211,238,0.2)]">
            <span className="w-2 h-2 rounded-full bg-emerald-400 dark:bg-emerald-400 light:bg-emerald-600 animate-ping" />
            <span className="text-xs font-mono font-bold tracking-wider text-cyan-300 dark:text-cyan-300 light:text-[#00639b] uppercase">
              Maritime India Vision 2030 & Deep Ocean Mission Platform
            </span>
          </div>

          {/* Main Title */}
          <h1 className="text-4xl sm:text-6xl lg:text-7xl font-black tracking-tight leading-tight max-w-5xl text-white dark:text-white light:text-slate-900">
            <span>EchoPulseNet:</span>{' '}
            <span className="bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400 dark:from-cyan-400 dark:via-teal-300 dark:to-emerald-400 light:from-[#00639b] light:via-teal-600 light:to-emerald-700 bg-clip-text text-transparent">
              AI-Powered Marine Sonar & Subsea Intelligence
            </span>
          </h1>

          {/* Subtitle */}
          <p className="text-base sm:text-lg lg:text-xl text-slate-300 dark:text-slate-300 light:text-slate-700 max-w-3xl font-normal leading-relaxed">
            An indigenous, autonomous acoustic intelligence platform empowering the{' '}
            <strong className="text-cyan-300 dark:text-cyan-300 light:text-[#00639b] font-bold">Government of India</strong> with real-time sonar target classification, 
            acoustic shadow physics, coastal defense surveillance, and marine ecological restoration.
          </p>

          {/* Action CTAs */}
          <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
            <GlassButton
              variant="primary"
              size="lg"
              onClick={() => navigate('/dashboard')}
              icon={<Radio className="w-5 h-5 animate-pulse text-cyan-300 dark:text-cyan-300 light:text-sky-800" />}
              className="px-8 py-3.5 text-sm font-bold tracking-wide shadow-[0_0_30px_rgba(34,211,238,0.4)]"
            >
              LAUNCH MISSION COMMAND & 3D TWIN
            </GlassButton>
            <GlassButton
              variant="secondary"
              size="lg"
              onClick={() => navigate('/sonar')}
              icon={<Waves className="w-5 h-5 text-emerald-400 dark:text-emerald-400 light:text-emerald-700" />}
              className="px-6 py-3.5 text-sm font-bold tracking-wide border-cyan-500/40"
            >
              SONAR WORKSTATION
            </GlassButton>
          </div>

          {/* Live System Spec Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 w-full max-w-4xl pt-8">
            <div className="p-3 rounded-2xl bg-[#030A17]/80 dark:bg-[#030A17]/80 light:bg-white border border-cyan-900/50 dark:border-cyan-900/50 light:border-sky-200 backdrop-blur-xl text-center shadow-sm">
              <div className="text-[11px] uppercase font-mono font-bold text-slate-400 dark:text-slate-400 light:text-slate-600">Target Latency</div>
              <div className="text-xl sm:text-2xl font-black text-cyan-300 dark:text-cyan-300 light:text-[#00639b] font-mono mt-0.5">3.4 ms</div>
              <div className="text-[10px] text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-semibold">~294 FPS Edge Inference</div>
            </div>
            <div className="p-3 rounded-2xl bg-[#030A17]/80 dark:bg-[#030A17]/80 light:bg-white border border-cyan-900/50 dark:border-cyan-900/50 light:border-sky-200 backdrop-blur-xl text-center shadow-sm">
              <div className="text-[11px] uppercase font-mono font-bold text-slate-400 dark:text-slate-400 light:text-slate-600">dGPU Acceleration</div>
              <div className="text-xl sm:text-2xl font-black text-emerald-300 dark:text-emerald-300 light:text-emerald-700 font-mono mt-0.5">RTX 5060</div>
              <div className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600">PyTorch 2.11 + CUDA 12.8</div>
            </div>
            <div className="p-3 rounded-2xl bg-[#030A17]/80 dark:bg-[#030A17]/80 light:bg-white border border-cyan-900/50 dark:border-cyan-900/50 light:border-sky-200 backdrop-blur-xl text-center shadow-sm">
              <div className="text-[11px] uppercase font-mono font-bold text-slate-400 dark:text-slate-400 light:text-slate-600">GIS Standard</div>
              <div className="text-xl sm:text-2xl font-black text-purple-300 dark:text-purple-300 light:text-[#60259e] font-mono mt-0.5">PostGIS</div>
              <div className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600">OGC GeoJSON / WGS84</div>
            </div>
            <div className="p-3 rounded-2xl bg-[#030A17]/80 dark:bg-[#030A17]/80 light:bg-white border border-cyan-900/50 dark:border-cyan-900/50 light:border-sky-200 backdrop-blur-xl text-center shadow-sm">
              <div className="text-[11px] uppercase font-mono font-bold text-slate-400 dark:text-slate-400 light:text-slate-600">Taxonomy</div>
              <div className="text-xl sm:text-2xl font-black text-amber-300 dark:text-amber-300 light:text-[#8a3b00] font-mono mt-0.5">8 Classes</div>
              <div className="text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-600">Hazards, Nets, UXO, Pipes</div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 1: STRATEGIC IMPORTANCE TO INDIAN GOVERNMENT */}
      <section className="py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200">
        <div className="text-center max-w-3xl mx-auto mb-10">
          <div className="flex items-center justify-center gap-2 text-cyan-400 dark:text-cyan-400 light:text-[#00639b] text-xs font-mono font-bold uppercase tracking-widest mb-2">
            <Building2 className="w-4 h-4" /> Sovereign Capabilities & National Security
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white dark:text-white light:text-slate-900">
            Strategic Alignment with Indian Government Initiatives
          </h2>
          <p className="text-sm sm:text-base text-slate-400 dark:text-slate-400 light:text-slate-600 mt-2">
            Engineered to support MoES, Indian Navy, Indian Coast Guard, and Sagarmala subsea infrastructure operations.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1 */}
          <GlassCard variant="default" className="p-6 space-y-4 hover:border-cyan-400/60 dark:hover:border-cyan-400/60 light:hover:border-sky-400 transition-all">
            <div className="w-12 h-12 rounded-2xl bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 border border-cyan-400/50 dark:border-cyan-400/50 light:border-sky-300 flex items-center justify-center text-cyan-300 dark:text-cyan-300 light:text-[#00639b] shadow-[0_0_15px_rgba(34,211,238,0.3)]">
              <Shield className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white dark:text-white light:text-slate-900">Deep Ocean Mission & MoES Alignment</h3>
            <p className="text-xs text-slate-300 dark:text-slate-300 light:text-slate-700 leading-relaxed">
              Provides sovereign, autonomous acoustic computer vision for Ministry of Earth Sciences (MoES) deep-sea exploration, 
              bathymetric seabed surveyance, and autonomous underwater vehicle (AUV/ROV) navigation in the Indian Ocean Exclusive Economic Zone (EEZ).
            </p>
            <div className="pt-2 border-t border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200 text-[11px] font-mono text-cyan-400 dark:text-cyan-400 light:text-[#00639b] font-semibold flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 dark:text-emerald-400 light:text-emerald-600" /> Samudrayaan & Matsya 6000 Telemetry Ready
            </div>
          </GlassCard>

          {/* Card 2 */}
          <GlassCard variant="default" className="p-6 space-y-4 hover:border-cyan-400/60 dark:hover:border-cyan-400/60 light:hover:border-sky-400 transition-all">
            <div className="w-12 h-12 rounded-2xl bg-emerald-500/20 dark:bg-emerald-500/20 light:bg-emerald-100 border border-emerald-400/50 dark:border-emerald-400/50 light:border-emerald-300 flex items-center justify-center text-emerald-300 dark:text-emerald-300 light:text-emerald-700 shadow-[0_0_15px_rgba(16,185,129,0.3)]">
              <Anchor className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white dark:text-white light:text-slate-900">Sagarmala & Critical Subsea Infrastructure</h3>
            <p className="text-xs text-slate-300 dark:text-slate-300 light:text-slate-700 leading-relaxed">
              Automated anomaly screening for high-voltage DC subsea power cables, offshore oil & gas pipelines (Mumbai High), 
              and harbor navigational channels. Detects anchor drag scars, free-spans, and structural scouring in real time.
            </p>
            <div className="pt-2 border-t border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200 text-[11px] font-mono text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-semibold flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 dark:text-emerald-400 light:text-emerald-600" /> Pipeline Scour & Anchor Drag Defense
            </div>
          </GlassCard>

          {/* Card 3 */}
          <GlassCard variant="default" className="p-6 space-y-4 hover:border-cyan-400/60 dark:hover:border-cyan-400/60 light:hover:border-sky-400 transition-all">
            <div className="w-12 h-12 rounded-2xl bg-purple-500/20 dark:bg-purple-500/20 light:bg-purple-100 border border-purple-400/50 dark:border-purple-400/50 light:border-purple-300 flex items-center justify-center text-purple-300 dark:text-purple-300 light:text-[#60259e] shadow-[0_0_15px_rgba(168,85,247,0.3)]">
              <Ship className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white dark:text-white light:text-slate-900">Coastal Defense & Mine Countermeasures (MCM)</h3>
            <p className="text-xs text-slate-300 dark:text-slate-300 light:text-slate-700 leading-relaxed">
              Enables naval hydrographic survey vessels to identify unexploded ordnance (UXO), underwater sea mines, submerged wrecks, 
              and illicit seabed installations with ray-grazing acoustic shadow verification and instant PostGIS spatial coordinates.
            </p>
            <div className="pt-2 border-t border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200 text-[11px] font-mono text-purple-400 dark:text-purple-400 light:text-[#60259e] font-semibold flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 dark:text-emerald-400 light:text-emerald-600" /> High-Confidence UXO & Hazard Pinpointing
            </div>
          </GlassCard>
        </div>
      </section>

      {/* SECTION 2: ECOLOGICAL & SOCIAL IMPACT */}
      <section className="py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200">
        <div className="text-center max-w-3xl mx-auto mb-10">
          <div className="flex items-center justify-center gap-2 text-emerald-400 dark:text-emerald-400 light:text-emerald-700 text-xs font-mono font-bold uppercase tracking-widest mb-2">
            <Fish className="w-4 h-4" /> Marine Conservation & Blue Economy
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white dark:text-white light:text-slate-900">
            Transformative Social & Ecological Impact
          </h2>
          <p className="text-sm sm:text-base text-slate-400 dark:text-slate-400 light:text-slate-600 mt-2">
            Protecting India’s 7,516 km coastline, artisanal fishing livelihoods, and endangered benthic reef ecosystems.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Ecological 1 */}
          <GlassCard variant="glow" className="p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-teal-500/20 dark:bg-teal-500/20 light:bg-teal-100 border border-teal-400/40 dark:border-teal-400/40 light:border-teal-300 flex items-center justify-center text-teal-300 dark:text-teal-300 light:text-teal-700">
                  <Waves className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-white dark:text-white light:text-slate-900">Eradication of Derelict Ghost Nets & Plastic Debris</h3>
              </div>
              <GlassBadge variant="emerald" size="sm">CRITICAL IMPACT</GlassBadge>
            </div>
            <p className="text-xs text-slate-300 dark:text-slate-300 light:text-slate-700 leading-relaxed">
              Abandoned fishing gear (ghost nets) traps endangered marine life—such as sea turtles, dugongs in the Gulf of Mannar, 
              and cetaceans—for decades while shedding toxic microplastics. EchoPulseNet's fine-tuned YOLOv12 model pinpoints submerged 
              nets with multi-factor confidence, delivering precise retrieval coordinates to automated grapple ROVs.
            </p>
            <div className="grid grid-cols-3 gap-2 pt-2 text-center text-xs font-mono">
              <div className="bg-[#030A17] dark:bg-[#030A17] light:bg-white p-2 rounded-xl border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200">
                <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold">100%</div>
                <div className="text-[9px] text-slate-400 dark:text-slate-400 light:text-slate-500">Ghost Net Auditing</div>
              </div>
              <div className="bg-[#030A17] dark:bg-[#030A17] light:bg-white p-2 rounded-xl border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200">
                <div className="text-cyan-300 dark:text-cyan-300 light:text-[#00639b] font-bold">&lt; 3.5m</div>
                <div className="text-[9px] text-slate-400 dark:text-slate-400 light:text-slate-500">GPS Geo-Accuracy</div>
              </div>
              <div className="bg-[#030A17] dark:bg-[#030A17] light:bg-white p-2 rounded-xl border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200">
                <div className="text-teal-300 dark:text-teal-300 light:text-teal-700 font-bold">Zero</div>
                <div className="text-[9px] text-slate-400 dark:text-slate-400 light:text-slate-500">Microplastic Loss</div>
              </div>
            </div>
          </GlassCard>

          {/* Ecological 2 */}
          <GlassCard variant="glow" className="p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-amber-500/20 dark:bg-amber-500/20 light:bg-amber-100 border border-amber-400/40 dark:border-amber-400/40 light:border-amber-300 flex items-center justify-center text-amber-300 dark:text-amber-300 light:text-[#8a3b00]">
                  <Globe2 className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-white dark:text-white light:text-slate-900">Empowering Indian Fisherfolk & Coastal Communities</h3>
              </div>
              <GlassBadge variant="cyan" size="sm">SOCIO-ECONOMIC</GlassBadge>
            </div>
            <p className="text-xs text-slate-300 dark:text-slate-300 light:text-slate-700 leading-relaxed">
              Subsea obstacles, snagged anchors, and submerged wrecks cause millions of rupees in gear loss, net tearing, and vessel damage 
              annually for coastal fisher communities across Tamil Nadu, Kerala, Gujarat, and Andhra Pradesh. EchoPulseNet automatically compiles 
              open-access navigational hazard maps to safeguard local fishing fleets.
            </p>
            <div className="grid grid-cols-3 gap-2 pt-2 text-center text-xs font-mono">
              <div className="bg-[#030A17] dark:bg-[#030A17] light:bg-white p-2 rounded-xl border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200">
                <div className="text-amber-300 dark:text-amber-300 light:text-[#8a3b00] font-bold">7,516 km</div>
                <div className="text-[9px] text-slate-400 dark:text-slate-400 light:text-slate-500">Coastline Coverage</div>
              </div>
              <div className="bg-[#030A17] dark:bg-[#030A17] light:bg-white p-2 rounded-xl border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200">
                <div className="text-cyan-300 dark:text-cyan-300 light:text-[#00639b] font-bold">4M+</div>
                <div className="text-[9px] text-slate-400 dark:text-slate-400 light:text-slate-500">Fisherfolk Protected</div>
              </div>
              <div className="bg-[#030A17] dark:bg-[#030A17] light:bg-white p-2 rounded-xl border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200">
                <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold">Automated</div>
                <div className="text-[9px] text-slate-400 dark:text-slate-400 light:text-slate-500">Navigational Warnings</div>
              </div>
            </div>
          </GlassCard>
        </div>
      </section>

      {/* SECTION 3: TECHNICAL ARCHITECTURE & DEPLOYMENT */}
      <section className="py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-sky-200">
        <div className="text-center max-w-3xl mx-auto mb-10">
          <div className="flex items-center justify-center gap-2 text-purple-400 dark:text-purple-400 light:text-[#60259e] text-xs font-mono font-bold uppercase tracking-widest mb-2">
            <Cpu className="w-4 h-4" /> Next-Generation Technology Stack
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white dark:text-white light:text-slate-900">
            Architecture Designed for Real-Time Edge Autonomy
          </h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-5 rounded-2xl bg-[#030A17]/90 dark:bg-[#030A17]/90 light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200 space-y-3 shadow-sm">
            <div className="text-cyan-400 dark:text-cyan-400 light:text-[#00639b] font-mono text-xs font-bold uppercase">1. Neural Inference Engine</div>
            <h4 className="text-sm font-bold text-white dark:text-white light:text-slate-900">YOLOv12 Attention-Centric (A2C2f)</h4>
            <p className="text-[11px] text-slate-400 dark:text-slate-400 light:text-slate-600 leading-relaxed">
              Fine-tuned on 1,972 acoustic sonar records with dGPU FP16 acceleration. Achieves 78.9% mAP@50 on critical pipeline anomalies.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-[#030A17]/90 dark:bg-[#030A17]/90 light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200 space-y-3 shadow-sm">
            <div className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-mono text-xs font-bold uppercase">2. Acoustic Ray Optics</div>
            <h4 className="text-sm font-bold text-white dark:text-white light:text-slate-900">Shadow Grazing Height Profiler</h4>
            <p className="text-[11px] text-slate-400 dark:text-slate-400 light:text-slate-600 leading-relaxed">
              Extracts 3D vertical object height from 2D sonar shadows based on sensor altitude and slant-range grazing geometry.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-[#030A17]/90 dark:bg-[#030A17]/90 light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200 space-y-3 shadow-sm">
            <div className="text-purple-400 dark:text-purple-400 light:text-[#60259e] font-mono text-xs font-bold uppercase">3. Spatial Intelligence</div>
            <h4 className="text-sm font-bold text-white dark:text-white light:text-slate-900">PostGIS & GeoAlchemy2</h4>
            <p className="text-[11px] text-slate-400 dark:text-slate-400 light:text-slate-600 leading-relaxed">
              Automatic transformation of sonar highlights to WGS84 GPS datum coordinates, supporting OGC GeoJSON, Shapefiles, and spatial radius queries.
            </p>
          </div>

          <div className="p-5 rounded-2xl bg-[#030A17]/90 dark:bg-[#030A17]/90 light:bg-white border border-cyan-900/40 dark:border-cyan-900/40 light:border-sky-200 space-y-3 shadow-sm">
            <div className="text-amber-400 dark:text-amber-400 light:text-[#8a3b00] font-mono text-xs font-bold uppercase">4. 3D Digital Twin</div>
            <h4 className="text-sm font-bold text-white dark:text-white light:text-slate-900">Interactive Bathymetry Surface</h4>
            <p className="text-[11px] text-slate-400 dark:text-slate-400 light:text-slate-600 leading-relaxed">
              Three.js WebGL digital twin rendering seafloor topography, AUV survey tracks, acoustic beam cones, and classified hazard markers.
            </p>
          </div>
        </div>

        {/* Bottom Launch Banner */}
        <div className="mt-12 p-8 rounded-3xl bg-gradient-to-r from-cyan-950/80 via-[#030B1B]/90 to-emerald-950/80 dark:from-cyan-950/80 dark:via-[#030B1B]/90 dark:to-emerald-950/80 light:from-sky-100 light:via-white light:to-emerald-50 border border-cyan-500/40 dark:border-cyan-500/40 light:border-sky-300 flex flex-col md:flex-row items-center justify-between gap-6 shadow-[0_0_40px_rgba(34,211,238,0.2)]">
          <div className="space-y-2 text-center md:text-left">
            <h3 className="text-2xl font-black text-white dark:text-white light:text-slate-900">Ready to Explore the Live Platform?</h3>
            <p className="text-xs sm:text-sm text-slate-300 dark:text-slate-300 light:text-slate-600 max-w-xl">
              Access the interactive command center, test live webcam computer vision, upload raw side-scan sonar files, or query spatial records.
            </p>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <GlassButton
              variant="primary"
              size="lg"
              onClick={() => navigate('/dashboard')}
              icon={<ArrowRight className="w-4 h-4" />}
              className="px-6 py-3 font-bold text-xs shadow-lg"
            >
              ENTER COMMAND CENTER
            </GlassButton>
          </div>
        </div>
      </section>
    </div>
  );
};
