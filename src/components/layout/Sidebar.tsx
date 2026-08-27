import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Home,
  LayoutDashboard,
  Radio,
  Crosshair,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Box,
  Camera,
  Shield,
} from 'lucide-react';

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapse: () => void;
}

interface NavItem {
  to: string;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  badge?: string;
  highlight?: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ collapsed, onToggleCollapse }) => {
  const navItems: NavItem[] = [
    { to: '/', icon: Home, label: 'OVERVIEW' },
    { to: '/dashboard', icon: LayoutDashboard, label: 'DASHBOARD' },
    { to: '/mpa', icon: Shield, label: 'MPA GEO-TAGS', badge: 'MoES', highlight: true },
    { to: '/upload', icon: Cpu, label: 'RAW INGESTION', badge: 'XTF/AI', highlight: true },
    { to: '/digital-twin', icon: Box, label: 'DIGITAL TWIN', badge: '3D', highlight: true },
    { to: '/webcam-tracker', icon: Camera, label: 'LIVE WEBCAM AI', badge: 'LIVE', highlight: true },
    { to: '/detections', icon: Crosshair, label: 'DETECTIONS' },
    { to: '/analytics', icon: BarChart3, label: 'INTELLIGENCE' },
  ];

  return (
    <aside
      className={`h-[calc(100vh-3.5rem)] bg-[#030A17]/80 dark:bg-[#030A17]/80 light:bg-white/85 backdrop-blur-2xl border-r border-cyan-500/25 dark:border-cyan-500/25 light:border-sky-300/60 flex flex-col justify-between transition-all duration-300 z-20 select-none shrink-0 shadow-[8px_0_32px_rgba(0,0,0,0.65),inset_1px_0_1.5px_rgba(255,255,255,0.22)] dark:shadow-[8px_0_32px_rgba(0,0,0,0.65),inset_1px_0_1.5px_rgba(255,255,255,0.22)] light:shadow-[4px_0_20px_rgba(15,23,42,0.06),inset_1px_0_1.5px_rgba(255,255,255,0.95)] relative overflow-hidden ${
        collapsed ? 'w-16' : 'w-56'
      }`}
    >
      {/* Top Liquid Specular Reflection */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-cyan-400/40 dark:via-cyan-400/40 light:via-sky-400/50 to-transparent pointer-events-none" />

      {/* Navigation list */}
      <div className="p-3 flex flex-col gap-1 overflow-y-auto overflow-x-hidden relative z-10">
        {!collapsed && (
          <div className="px-3 pb-2 text-[10px] uppercase font-mono font-bold tracking-widest text-slate-400 dark:text-slate-400 light:text-slate-500">
            SYSTEM MODULES
          </div>
        )}

        <div className="space-y-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 text-xs font-medium rounded-xl transition-all duration-200 relative overflow-hidden backdrop-blur-xl ${
                    isActive
                      ? 'bg-gradient-to-b from-cyan-500/30 to-cyan-600/15 dark:from-cyan-500/30 dark:to-cyan-600/15 light:from-sky-100 light:to-sky-200 text-cyan-200 dark:text-cyan-200 light:text-sky-900 font-bold border border-cyan-400/60 dark:border-cyan-400/60 light:border-sky-400 shadow-[0_0_20px_rgba(34,211,238,0.25),inset_0_1px_1px_rgba(255,255,255,0.4)] dark:shadow-[0_0_20px_rgba(34,211,238,0.25),inset_0_1px_1px_rgba(255,255,255,0.4)] light:shadow-[0_2px_8px_rgba(2,132,199,0.15),inset_0_1px_1px_rgba(255,255,255,0.9)]'
                      : 'text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-slate-100 dark:hover:text-slate-100 light:hover:text-slate-900 hover:bg-cyan-950/40 dark:hover:bg-cyan-950/40 light:hover:bg-slate-100/90 border border-transparent'
                  } ${collapsed ? 'justify-center px-0' : ''}`
                }
                title={collapsed ? item.label : undefined}
              >
                <Icon
                  className={`w-4 h-4 shrink-0 ${
                    item.highlight
                      ? 'text-cyan-400 dark:text-cyan-400 light:text-sky-600 animate-pulse'
                      : ''
                  }`}
                />
                {!collapsed && (
                  <div className="flex items-center justify-between flex-1 truncate font-mono">
                    <span className="tracking-wider text-[11px] uppercase truncate">{item.label}</span>
                    {item.badge && (
                      <span
                        className={`text-[9px] px-2 py-0.2 rounded-full font-mono font-bold ${
                          item.highlight
                            ? 'bg-cyan-400 dark:bg-cyan-400 light:bg-sky-600 text-slate-950 dark:text-slate-950 light:text-white font-black shadow-[0_0_8px_rgba(34,211,238,0.4)]'
                            : 'bg-cyan-950/90 dark:bg-cyan-950/90 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-sky-800 border border-cyan-500/50 dark:border-cyan-500/50 light:border-sky-300'
                        }`}
                      >
                        {item.badge}
                      </span>
                    )}
                  </div>
                )}
              </NavLink>
            );
          })}
        </div>
      </div>

      {/* Engine Status & Footer Collapse Toggle */}
      <div className="p-3 border-t border-cyan-900/40 dark:border-cyan-900/40 light:border-slate-200/80 flex flex-col gap-2 relative z-10">
        {!collapsed && (
          <div className="bg-[#020814]/75 dark:bg-[#020814]/75 light:bg-slate-100/90 backdrop-blur-md p-2.5 rounded-xl border border-cyan-900/45 dark:border-cyan-900/45 light:border-slate-300 text-[10px] font-mono shadow-inner">
            <div className="flex justify-between items-center mb-1">
              <span className="text-slate-400 dark:text-slate-400 light:text-slate-600 uppercase tracking-wider text-[9px] flex items-center gap-1.5 font-bold">
                <Cpu className="w-3 h-3 text-cyan-400 dark:text-cyan-400 light:text-sky-600" />
                Inference Core
              </span>
              <span className="flex items-center gap-1 text-[9px] text-emerald-400 dark:text-emerald-400 light:text-emerald-700 font-bold">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 dark:bg-emerald-400 light:bg-emerald-600 animate-pulse shadow-[0_0_6px_#34d399]" />
                ONLINE
              </span>
            </div>
            <div className="text-slate-400 dark:text-slate-400 light:text-slate-600 text-[9px] truncate font-medium">
              HydroPhys-OmniNet (CAW-SSM)
            </div>
          </div>
        )}

        <div className="flex items-center justify-between pt-1">
          {!collapsed && (
            <span className="text-[9px] text-slate-500 dark:text-slate-500 light:text-slate-400 font-mono tracking-wider font-semibold">
              ECHOPULSENET SIH26057
            </span>
          )}
          <button
            onClick={onToggleCollapse}
            className="p-1.5 rounded-lg hover:bg-cyan-950/50 dark:hover:bg-cyan-950/50 light:hover:bg-slate-200 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-700 transition-colors mx-auto"
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>
      </div>
    </aside>
  );
};
