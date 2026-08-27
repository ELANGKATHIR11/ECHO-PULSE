import React, { useState, useEffect } from 'react';
import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import {
  Activity,
  Cpu,
  Radio,
  Zap,
  Box,
  Compass,
  Sun,
  Moon,
  Shield,
} from 'lucide-react';
import { systemApi } from '../../services/systemApi';
import { missionApi } from '../../services/missionApi';
import { SystemTelemetry, Mission } from '../../types';
import { GlassBadge, GlassButton } from '../glass/GlassCard';
import { useTheme } from '../../context/ThemeContext';

export const Navbar: React.FC = () => {
  const { theme, isDark, toggleTheme } = useTheme();
  const [telemetry, setTelemetry] = useState<SystemTelemetry | null>(null);
  const [missions, setMissions] = useState<Mission[]>([]);
  const [activeMissionId, setActiveMissionId] = useState<string>('MSN-2025-08-01');
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    missionApi.getMissions().then((data) => {
      setMissions(data);
      if (data.length > 0) setActiveMissionId(data[0].id);
    });

    const fetchTelem = async () => {
      try {
        const data = await systemApi.getTelemetry();
        setTelemetry(data);
      } catch {}
    };
    fetchTelem();
    const interval = setInterval(fetchTelem, 3000);
    return () => clearInterval(interval);
  }, []);

  const activeMission = missions.find((m) => m.id === activeMissionId);

  return (
    <header className="h-14 bg-[#030A17]/85 dark:bg-[#030A17]/85 light:bg-white/90 backdrop-blur-2xl border-b border-cyan-500/25 dark:border-cyan-500/25 light:border-sky-300/60 px-3 sm:px-4 flex items-center justify-between gap-2 sm:gap-4 z-40 font-mono shadow-[0_8px_32px_rgba(0,0,0,0.7),inset_0_1px_1.5px_rgba(255,255,255,0.25)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.7),inset_0_1px_1.5px_rgba(255,255,255,0.25)] light:shadow-[0_4px_20px_rgba(15,23,42,0.06),inset_0_1px_1.5px_rgba(255,255,255,0.95)] select-none transition-all relative overflow-hidden">
      {/* Top Liquid Specular Reflection Edge */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-cyan-400/40 dark:via-cyan-400/40 light:via-sky-400/60 to-transparent pointer-events-none" />

      {/* Brand & Survey Selector */}
      <div className="flex items-center gap-2 sm:gap-3 min-w-0 shrink-0">
        <NavLink
          to="/"
          className="flex items-center gap-2 text-white dark:text-white light:text-slate-900 hover:opacity-95 transition-opacity shrink-0"
        >
          <div className="w-8 h-8 rounded-xl bg-gradient-to-b from-cyan-500/30 to-cyan-700/20 dark:from-cyan-500/30 dark:to-cyan-700/20 light:from-sky-100 light:to-sky-200 border border-cyan-400/60 dark:border-cyan-400/60 light:border-sky-400 flex items-center justify-center shadow-[0_0_16px_rgba(34,211,238,0.35)] shrink-0">
            <Radio className="w-4 h-4 text-cyan-400 dark:text-cyan-400 light:text-sky-600 animate-pulse" />
          </div>
          <div className="hidden sm:block">
            <div className="text-xs sm:text-sm font-black tracking-widest leading-none flex items-center gap-1 text-white dark:text-white light:text-slate-900">
              ECHOPULSENET
              <span className="text-[8px] px-1 py-0.5 rounded-full bg-cyan-500/25 dark:bg-cyan-500/25 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-sky-700 border border-cyan-400/50 font-bold">
                PRO
              </span>
            </div>
            <div className="text-[8px] text-cyan-400/90 dark:text-cyan-400/90 light:text-sky-700 font-bold tracking-widest mt-0.5">
              LIQUID DIGITAL TWIN
            </div>
          </div>
        </NavLink>

        {/* Active Survey Mission Selector */}
        <div className="hidden xl:flex items-center gap-1.5 bg-[#020814]/70 dark:bg-[#020814]/70 light:bg-slate-100/90 backdrop-blur-md border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-300 px-2.5 py-1 rounded-xl text-xs shadow-inner max-w-[240px] 2xl:max-w-[320px] shrink-0">
          <Compass className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600 shrink-0" />
          <span className="text-slate-400 dark:text-slate-400 light:text-slate-500 text-[9px] uppercase font-bold shrink-0">
            SURVEY:
          </span>
          <select
            value={activeMissionId}
            onChange={(e) => setActiveMissionId(e.target.value)}
            className="bg-transparent text-cyan-300 dark:text-cyan-300 light:text-sky-800 font-bold focus:outline-none text-[11px] cursor-pointer truncate max-w-[140px] 2xl:max-w-[200px]"
          >
            {missions.map((m) => (
              <option key={m.id} value={m.id} className="bg-[#040D1B] dark:bg-[#040D1B] light:bg-white text-slate-200 dark:text-slate-200 light:text-slate-800">
                {m.name}
              </option>
            ))}
          </select>
          {activeMission && (
            <span className="text-[9px] text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold px-1.5 py-0.5 rounded-full bg-emerald-950/70 border border-emerald-500/40 shrink-0">
              {activeMission.status}
            </span>
          )}
        </div>
      </div>

      {/* GPU & Neural Engine Telemetry Strip */}
      <div className="hidden 2xl:flex items-center gap-2.5 text-xs text-slate-300 shrink-0">
        {/* NVIDIA RTX status */}
        <div className="flex items-center gap-1.5 bg-[#020814]/60 dark:bg-[#020814]/60 light:bg-slate-100/90 px-2.5 py-1 rounded-xl border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-300 shadow-inner">
          <Cpu className="w-3.5 h-3.5 text-emerald-400 dark:text-emerald-400 light:text-emerald-600" />
          <span className="text-slate-400 text-[10px]">GPU:</span>
          <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold">
            {telemetry && telemetry.gpuUtilPct !== null && telemetry.gpuUtilPct !== undefined ? `${telemetry.gpuUtilPct}%` : 'ONLINE'}
          </span>
          <span className="text-slate-500 text-[10px]">
            ({telemetry?.vramUsedGb ? `${telemetry.vramUsedGb}GB` : 'RTX 5060'})
          </span>
        </div>

        {/* Neural Pipeline Model */}
        <div className="flex items-center gap-1.5 bg-[#020814]/60 dark:bg-[#020814]/60 light:bg-slate-100/90 px-2.5 py-1 rounded-xl border border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-300 shadow-inner">
          <Zap className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
          <span className="text-slate-400 text-[10px]">MODEL:</span>
          <span className="text-cyan-300 dark:text-cyan-300 light:text-sky-700 font-bold">EchoPhys-X</span>
          <span className="text-slate-500 text-[10px]">
            {telemetry?.inferenceFps ? `${telemetry.inferenceFps} FPS` : '185 FPS'}
          </span>
        </div>

        {/* System Health Status */}
        <GlassBadge variant="emerald" size="sm" pulse>
          SYSTEM NOMINAL
        </GlassBadge>
      </div>

      {/* Quick Action Navigation Buttons & Theme Toggle */}
      <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
        {/* Light/Dark Mode Switcher */}
        <button
          onClick={toggleTheme}
          aria-label={isDark ? 'Switch to Bright Light Mode' : 'Switch to Dark Mode'}
          className={`flex items-center gap-1 px-2.5 py-1 rounded-xl border text-[11px] font-mono font-bold transition-all relative overflow-hidden backdrop-blur-xl active:scale-95 shrink-0 ${
            isDark
              ? 'bg-gradient-to-b from-amber-500/25 to-amber-600/10 border-amber-400/50 text-amber-300 hover:border-amber-300 shadow-[0_0_16px_rgba(245,158,11,0.25)]'
              : 'bg-gradient-to-b from-sky-100 to-sky-200 border-sky-300 text-sky-900 hover:border-sky-400 shadow-[0_2px_8px_rgba(2,132,199,0.15)]'
          }`}
          title={isDark ? 'Activate Bright Light Mode' : 'Activate Dark Mode'}
        >
          {isDark ? (
            <>
              <Sun className="w-3.5 h-3.5 text-amber-400 animate-spin-slow" />
              <span className="hidden md:inline">LIGHT MODE</span>
            </>
          ) : (
            <>
              <Moon className="w-3.5 h-3.5 text-sky-700" />
              <span className="hidden md:inline">DARK MODE</span>
            </>
          )}
        </button>

        {/* Tactical Defense Command Center Button */}
        <GlassButton
          variant={location.pathname === '/command-center' ? 'primary' : 'secondary'}
          size="sm"
          icon={<Radio className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />}
          onClick={() => navigate('/command-center')}
          className="text-[11px] font-bold px-2.5 py-1 shrink-0"
        >
          <span className="hidden sm:inline">COMMAND HUD</span>
        </GlassButton>

        {/* 3D Digital Twin Quick Access */}
        <GlassButton
          variant={location.pathname === '/digital-twin' ? 'primary' : 'secondary'}
          size="sm"
          icon={<Box className="w-3.5 h-3.5 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />}
          onClick={() => navigate('/digital-twin')}
          className="text-[11px] px-2.5 py-1 shrink-0"
        >
          <span className="hidden sm:inline">DIGITAL TWIN</span>
        </GlassButton>

        {/* Indian MPA Zones Quick Access */}
        <GlassButton
          variant={location.pathname === '/mpa' ? 'primary' : 'secondary'}
          size="sm"
          icon={<Shield className="w-3.5 h-3.5 text-emerald-400 dark:text-emerald-400 light:text-emerald-600" />}
          onClick={() => navigate('/mpa')}
          className="text-[11px] px-2.5 py-1 shrink-0"
        >
          <span className="hidden sm:inline">MPA ZONES</span>
        </GlassButton>
      </div>
    </header>
  );
};

