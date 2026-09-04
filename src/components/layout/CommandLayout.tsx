import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Navbar } from './Navbar';
import { Sidebar } from './Sidebar';
import { RenderProfile } from '../../types';
import { OceanLiquidCausticBackground } from '../glass/OceanLiquidCausticBackground';

export const CommandLayout: React.FC = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [renderProfile] = useState<RenderProfile>('HIGH');
  const [activeMissionName, setActiveMissionName] = useState('HYDRO-SCAN ALPHA-07 (MSN-0884)');

  return (
    <div className="h-screen w-screen bg-[#020712] dark:bg-[#020712] light:bg-[#d2ecf9] text-slate-200 dark:text-slate-200 light:text-[#061930] font-sans flex flex-col overflow-hidden select-none relative transition-colors">
      {/* 3D Ocean-Blue Translucent Liquid-Glass Caustic Engine & Interactive Hydro-Particles */}
      <OceanLiquidCausticBackground interactive={true} />

      {/* Background Liquid Caustic Drift Texture */}
      <div className="liquid-caustic-layer" />

      {/* Top Liquid Command Header */}
      <Navbar />

      {/* Main App Body */}
      <div className="flex flex-1 overflow-hidden relative z-10">
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        />

        {/* Content Viewport */}
        <main className="flex-1 overflow-y-auto bg-transparent relative flex flex-col transition-colors">
          <Outlet context={{ renderProfile, activeMissionName, setActiveMissionName }} />
        </main>
      </div>

      {/* Sleek Liquid Glass Status Footer */}
      <footer className="h-6 bg-[#030914]/90 dark:bg-[#030914]/90 light:bg-white/90 backdrop-blur-xl border-t border-cyan-500/20 dark:border-cyan-500/20 light:border-sky-200 flex items-center px-4 justify-between text-[9px] font-mono text-slate-400 dark:text-slate-400 light:text-slate-600 uppercase tracking-widest shrink-0 z-30 transition-colors shadow-[0_-4px_16px_rgba(0,0,0,0.4)]">
        <div className="flex gap-4 items-center">
          <span className="hidden sm:inline">Network: Encrypted (AES-256-GCM)</span>
          <span className="text-cyan-400/80 dark:text-cyan-400/80 light:text-sky-700">HydroLink: Active</span>
          <span className="hidden md:inline text-slate-500">srv-01.epn.marine.gov</span>
        </div>
        <div className="flex gap-4 items-center">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 dark:bg-emerald-400 light:bg-emerald-600 shadow-[0_0_6px_#34d399]" />
            <span className="text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold">
              Digital Twin Liquid Engine
            </span>
          </div>
          <span className="hidden md:inline">(c) 2026 EchoPulseNet AI • SIH26057</span>
        </div>
      </footer>
    </div>
  );
};

