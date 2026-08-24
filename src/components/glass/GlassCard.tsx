import React from 'react';

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  variant?: 'default' | 'subtle' | 'glow' | 'interactive' | 'deep' | 'kpi';
  className?: string;
  liquid?: boolean;
}

export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  variant = 'default',
  className = '',
  liquid = true,
  ...props
}) => {
  if (variant === 'kpi') {
    return (
      <div
        className={`kpi-card relative overflow-hidden transition-all duration-300 ${className}`}
        {...props}
      >
        {/* Liquid Specular Top Sheen Edge */}
        <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-cyan-400/40 dark:via-cyan-400/40 light:via-sky-400/50 to-transparent pointer-events-none z-10" />

        {/* Convex Ambient Flare */}
        {liquid && (
          <div className="absolute top-0 left-0 w-36 h-24 bg-radial from-white/[0.08] dark:from-white/[0.08] light:from-white/[0.6] to-transparent pointer-events-none z-0" />
        )}

        <div className="relative z-10 flex flex-col justify-between flex-1">{children}</div>
      </div>
    );
  }

  const variantStyles = {
    default:
      'bg-[#050E1F]/70 dark:bg-[#050E1F]/70 light:bg-white/85 backdrop-blur-2xl border border-cyan-500/25 dark:border-cyan-500/25 light:border-sky-300/60 shadow-[0_16px_40px_-10px_rgba(0,0,0,0.75),inset_0_1px_1.5px_0_rgba(255,255,255,0.22)] dark:shadow-[0_16px_40px_-10px_rgba(0,0,0,0.75),inset_0_1px_1.5px_0_rgba(255,255,255,0.22)] light:shadow-[0_10px_30px_-6px_rgba(15,23,42,0.08),inset_0_1px_1.5px_0_rgba(255,255,255,0.95)] text-slate-200 dark:text-slate-200 light:text-slate-800',
    subtle:
      'bg-[#040B18]/55 dark:bg-[#040B18]/55 light:bg-slate-50/75 backdrop-blur-xl border border-cyan-900/35 dark:border-cyan-900/35 light:border-slate-200/90 shadow-[0_8px_24px_rgba(0,0,0,0.45),inset_0_1px_1px_0_rgba(255,255,255,0.15)] dark:shadow-[0_8px_24px_rgba(0,0,0,0.45),inset_0_1px_1px_0_rgba(255,255,255,0.15)] light:shadow-[0_4px_16px_rgba(15,23,42,0.04),inset_0_1px_1px_0_rgba(255,255,255,0.9)] text-slate-300 dark:text-slate-300 light:text-slate-700',
    glow:
      'bg-[#051124]/80 dark:bg-[#051124]/80 light:bg-white/92 backdrop-blur-2xl border border-cyan-400/50 dark:border-cyan-400/50 light:border-sky-400/70 shadow-[0_0_30px_rgba(6,182,212,0.22),0_16px_48px_rgba(0,0,0,0.8),inset_0_1px_2px_0_rgba(255,255,255,0.35)] dark:shadow-[0_0_30px_rgba(6,182,212,0.22),0_16px_48px_rgba(0,0,0,0.8),inset_0_1px_2px_0_rgba(255,255,255,0.35)] light:shadow-[0_0_24px_rgba(2,132,199,0.16),0_12px_36px_rgba(15,23,42,0.09),inset_0_1px_2px_0_rgba(255,255,255,1)] text-white dark:text-white light:text-slate-900',
    interactive:
      'bg-[#050E1F]/70 dark:bg-[#050E1F]/70 light:bg-white/85 backdrop-blur-2xl border border-cyan-500/25 dark:border-cyan-500/25 light:border-sky-300/60 hover:border-cyan-400/60 dark:hover:border-cyan-400/60 light:hover:border-sky-500/80 hover:bg-[#081830]/85 dark:hover:bg-[#081830]/85 light:hover:bg-sky-50/95 transition-all duration-300 hover:shadow-[0_0_24px_rgba(34,211,238,0.25),0_16px_40px_rgba(0,0,0,0.7),inset_0_1px_2px_0_rgba(255,255,255,0.4)] dark:hover:shadow-[0_0_24px_rgba(34,211,238,0.25),0_16px_40px_rgba(0,0,0,0.7),inset_0_1px_2px_0_rgba(255,255,255,0.4)] light:hover:shadow-[0_8px_28px_rgba(2,132,199,0.16),inset_0_1px_2px_0_rgba(255,255,255,1)] cursor-pointer text-slate-200 dark:text-slate-200 light:text-slate-800 active:scale-[0.99]',
    deep:
      'bg-[#020610]/85 dark:bg-[#020610]/85 light:bg-slate-100/90 backdrop-blur-2xl border border-cyan-900/45 dark:border-cyan-900/45 light:border-slate-300/80 shadow-[0_20px_50px_rgba(0,0,0,0.85),inset_0_1px_1.5px_0_rgba(255,255,255,0.18)] dark:shadow-[0_20px_50px_rgba(0,0,0,0.85),inset_0_1px_1.5px_0_rgba(255,255,255,0.18)] light:shadow-[0_8px_30px_rgba(15,23,42,0.06),inset_0_1px_1.5px_0_rgba(255,255,255,0.85)] text-slate-200 dark:text-slate-200 light:text-slate-800',
  };

  return (
    <div
      className={`rounded-2xl relative overflow-hidden transition-all duration-300 ${variantStyles[variant]} ${className}`}
      {...props}
    >
      {/* Liquid Specular Top Sheen Edge */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-cyan-400/40 dark:via-cyan-400/40 light:via-sky-400/50 to-transparent pointer-events-none z-10" />

      {/* Convex Ambient Flare */}
      {liquid && (
        <div className="absolute top-0 left-0 w-36 h-24 bg-radial from-white/[0.08] dark:from-white/[0.08] light:from-white/[0.6] to-transparent pointer-events-none z-0" />
      )}

      <div className="relative z-10">{children}</div>
    </div>
  );
};

interface GlassPanelProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  header?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}

export const GlassPanel: React.FC<GlassPanelProps> = ({
  children,
  header,
  actions,
  className = '',
  bodyClassName = 'p-4',
  ...props
}) => {
  return (
    <div
      className={`rounded-2xl bg-[#040E1E]/75 dark:bg-[#040E1E]/75 light:bg-white/85 backdrop-blur-2xl border border-cyan-500/25 dark:border-cyan-500/25 light:border-sky-300/60 shadow-[0_16px_40px_-10px_rgba(0,0,0,0.75),inset_0_1px_1.5px_0_rgba(255,255,255,0.25)] dark:shadow-[0_16px_40px_-10px_rgba(0,0,0,0.75),inset_0_1px_1.5px_0_rgba(255,255,255,0.25)] light:shadow-[0_10px_30px_-6px_rgba(15,23,42,0.08),inset_0_1px_1.5px_0_rgba(255,255,255,0.95)] flex flex-col relative overflow-hidden text-slate-200 dark:text-slate-200 light:text-slate-800 transition-all duration-300 ${className}`}
      {...props}
    >
      {/* Liquid Specular Top Sheen */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-cyan-400/40 dark:via-cyan-400/40 light:via-sky-400/50 to-transparent pointer-events-none z-10" />

      {/* Convex Ambient Flare */}
      <div className="absolute top-0 left-0 w-44 h-24 bg-radial from-white/[0.08] dark:from-white/[0.08] light:from-white/[0.6] to-transparent pointer-events-none z-0" />

      {header && (
        <div className="px-4 py-3 border-b border-cyan-900/35 dark:border-cyan-900/35 light:border-slate-200/80 bg-[#020814]/60 dark:bg-[#020814]/60 light:bg-slate-50/80 backdrop-blur-md flex items-center justify-between gap-2 shrink-0 relative z-10">
          <div className="text-xs font-mono font-bold tracking-wider text-slate-200 dark:text-slate-200 light:text-slate-800 uppercase flex items-center gap-2">
            {header}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className={`flex-1 overflow-auto relative z-10 ${bodyClassName}`}>{children}</div>
    </div>
  );
};

interface GlassButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'amber';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
  className?: string;
}

export const GlassButton: React.FC<GlassButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  icon,
  className = '',
  disabled,
  ...props
}) => {
  const sizeStyles = {
    sm: 'px-3 py-1 text-xs gap-1.5 min-h-[32px] rounded-lg',
    md: 'px-4 py-1.5 text-xs gap-2 min-h-[38px] rounded-xl',
    lg: 'px-5 py-2.5 text-sm gap-2.5 min-h-[44px] rounded-xl',
  };

  const variantStyles = {
    primary:
      'bg-gradient-to-b from-cyan-500/30 to-cyan-600/15 dark:from-cyan-500/30 dark:to-cyan-600/15 light:from-sky-500 light:to-sky-600 hover:from-cyan-400/40 hover:to-cyan-500/25 dark:hover:from-cyan-400/40 dark:hover:to-cyan-500/25 light:hover:from-sky-600 light:hover:to-sky-700 text-cyan-200 dark:text-cyan-200 light:text-white border border-cyan-400/60 dark:border-cyan-400/60 light:border-sky-400 shadow-[0_0_20px_rgba(34,211,238,0.25),inset_0_1px_1px_rgba(255,255,255,0.4)] dark:shadow-[0_0_20px_rgba(34,211,238,0.25),inset_0_1px_1px_rgba(255,255,255,0.4)] light:shadow-[0_4px_14px_rgba(2,132,199,0.35),inset_0_1px_1px_rgba(255,255,255,0.8)] font-bold',
    secondary:
      'bg-[#061528]/80 dark:bg-[#061528]/80 light:bg-white/90 hover:bg-cyan-950/60 dark:hover:bg-cyan-950/60 light:hover:bg-slate-100 text-slate-300 dark:text-slate-300 light:text-slate-700 hover:text-white dark:hover:text-white light:hover:text-slate-900 border border-cyan-900/50 dark:border-cyan-900/50 light:border-slate-300 shadow-[0_4px_12px_rgba(0,0,0,0.3),inset_0_1px_1px_rgba(255,255,255,0.15)] dark:shadow-[0_4px_12px_rgba(0,0,0,0.3),inset_0_1px_1px_rgba(255,255,255,0.15)] light:shadow-[0_2px_8px_rgba(15,23,42,0.06),inset_0_1px_1px_rgba(255,255,255,0.9)] font-medium',
    ghost:
      'bg-transparent hover:bg-cyan-500/15 dark:hover:bg-cyan-500/15 light:hover:bg-sky-100/80 text-slate-400 dark:text-slate-400 light:text-slate-600 hover:text-cyan-300 dark:hover:text-cyan-300 light:hover:text-sky-800 border border-transparent font-medium',
    danger:
      'bg-gradient-to-b from-rose-500/25 to-rose-600/15 dark:from-rose-500/25 dark:to-rose-600/15 light:from-rose-500 light:to-rose-600 hover:from-rose-500/35 hover:to-rose-600/25 dark:hover:from-rose-500/35 dark:hover:to-rose-600/25 light:hover:from-rose-600 light:hover:to-rose-700 text-rose-200 dark:text-rose-200 light:text-white border border-rose-500/60 dark:border-rose-500/60 light:border-rose-400 shadow-[0_0_16px_rgba(244,63,94,0.25),inset_0_1px_1px_rgba(255,255,255,0.3)] font-bold',
    amber:
      'bg-gradient-to-b from-amber-500/30 to-amber-600/15 dark:from-amber-500/30 dark:to-amber-600/15 light:from-amber-500 light:to-amber-600 hover:from-amber-500/40 hover:to-amber-600/25 dark:hover:from-amber-500/40 dark:hover:to-amber-600/25 light:hover:from-amber-600 light:hover:to-amber-700 text-amber-200 dark:text-amber-200 light:text-white border border-amber-500/60 dark:border-amber-500/60 light:border-amber-400 shadow-[0_0_16px_rgba(245,158,11,0.25),inset_0_1px_1px_rgba(255,255,255,0.4)] font-bold',
  };

  return (
    <button
      disabled={disabled}
      className={`inline-flex items-center justify-center font-mono transition-all duration-200 select-none whitespace-nowrap outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 active:scale-[0.97] disabled:opacity-45 disabled:pointer-events-none relative overflow-hidden backdrop-blur-xl ${sizeStyles[size]} ${variantStyles[variant]} ${className}`}
      {...props}
    >
      {/* Liquid Top Refraction Line */}
      <span className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-white/40 to-transparent pointer-events-none" />
      {icon && <span className="shrink-0 relative z-10">{icon}</span>}
      <span className="relative z-10">{children}</span>
    </button>
  );
};

interface GlassStatProps {
  label: string;
  value: string | number;
  unit?: string;
  trend?: string;
  trendPositive?: boolean;
  icon?: React.ReactNode;
  subtext?: string;
  variant?: 'cyan' | 'emerald' | 'amber' | 'purple' | 'rose';
  className?: string;
}

export const GlassStat: React.FC<GlassStatProps> = ({
  label,
  value,
  unit,
  trend,
  trendPositive,
  icon,
  subtext,
  variant = 'cyan',
  className = '',
}) => {
  const accentColors = {
    cyan: 'text-cyan-400 dark:text-cyan-400 light:text-[#00639b] border-cyan-500/35 dark:border-cyan-500/35 light:border-sky-300 bg-cyan-500/15 dark:bg-cyan-500/15 light:bg-sky-100 shadow-[0_0_12px_rgba(34,211,238,0.2)]',
    emerald:
      'text-emerald-400 dark:text-emerald-400 light:text-[#03624c] border-emerald-500/35 dark:border-emerald-500/35 light:border-emerald-300 bg-emerald-500/15 dark:bg-emerald-500/15 light:bg-emerald-100 shadow-[0_0_12px_rgba(16,185,129,0.2)]',
    amber:
      'text-amber-400 dark:text-amber-400 light:text-[#8a3b00] border-amber-500/35 dark:border-amber-500/35 light:border-amber-300 bg-amber-500/15 dark:bg-amber-500/15 light:bg-amber-100 shadow-[0_0_12px_rgba(245,158,11,0.2)]',
    purple:
      'text-purple-400 dark:text-purple-400 light:text-[#60259e] border-purple-500/35 dark:border-purple-500/35 light:border-purple-300 bg-purple-500/15 dark:bg-purple-500/15 light:bg-purple-100 shadow-[0_0_12px_rgba(168,85,247,0.2)]',
    rose: 'text-rose-400 dark:text-rose-400 light:text-[#9e1030] border-rose-500/35 dark:border-rose-500/35 light:border-rose-300 bg-rose-500/15 dark:bg-rose-500/15 light:bg-rose-100 shadow-[0_0_12px_rgba(244,63,94,0.2)]',
  };

  return (
    <div
      className={`rounded-2xl bg-[#040E1E]/75 dark:bg-[#040E1E]/75 light:bg-white/85 backdrop-blur-2xl border border-cyan-500/25 dark:border-cyan-500/25 light:border-sky-300/60 p-4 shadow-[0_16px_40px_-10px_rgba(0,0,0,0.7),inset_0_1px_1.5px_0_rgba(255,255,255,0.22)] dark:shadow-[0_16px_40px_-10px_rgba(0,0,0,0.7),inset_0_1px_1.5px_0_rgba(255,255,255,0.22)] light:shadow-[0_8px_24px_rgba(15,23,42,0.06),inset_0_1px_1.5px_0_rgba(255,255,255,0.95)] flex flex-col justify-between font-mono relative overflow-hidden transition-all duration-300 hover:border-cyan-400/50 dark:hover:border-cyan-400/50 light:hover:border-sky-400 hover:translate-y-[-1px] ${className}`}
    >
      {/* Liquid Specular Top Sheen */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-cyan-400/30 dark:via-cyan-400/30 light:via-sky-400/40 to-transparent pointer-events-none" />

      {/* Ambient Flare */}
      <div className="absolute top-0 left-0 w-32 h-20 bg-radial from-white/[0.07] dark:from-white/[0.07] light:from-white/[0.5] to-transparent pointer-events-none" />

      <div className="flex items-center justify-between mb-2 relative z-10">
        <span className="text-[10px] uppercase font-bold tracking-widest text-slate-400 dark:text-slate-400 light:text-slate-500 truncate">
          {label}
        </span>
        {icon && (
          <div
            className={`w-7 h-7 rounded-lg flex items-center justify-center border backdrop-blur-md ${accentColors[variant]}`}
          >
            {icon}
          </div>
        )}
      </div>

      <div className="flex items-baseline gap-1.5 my-0.5 relative z-10">
        <span className="text-2xl font-black text-white dark:text-white light:text-slate-900 tracking-tight">
          {value}
        </span>
        {unit && (
          <span className="text-xs text-slate-400 dark:text-slate-400 light:text-slate-500 font-semibold">
            {unit}
          </span>
        )}
      </div>

      {(trend || subtext) && (
        <div className="flex items-center justify-between text-[10px] text-slate-400 dark:text-slate-400 light:text-slate-500 mt-2 pt-1.5 border-t border-cyan-900/30 dark:border-cyan-900/30 light:border-slate-200/80 relative z-10">
          {subtext && <span className="truncate">{subtext}</span>}
          {trend && (
            <span
              className={`font-bold ml-auto ${
                trendPositive
                  ? 'text-emerald-400 dark:text-emerald-400 light:text-emerald-600'
                  : 'text-rose-400 dark:text-rose-400 light:text-rose-600'
              }`}
            >
              {trend}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

interface GlassBadgeProps {
  children: React.ReactNode;
  variant?: 'cyan' | 'emerald' | 'amber' | 'rose' | 'purple' | 'slate';
  size?: 'sm' | 'md';
  pulse?: boolean;
  className?: string;
}

export const GlassBadge: React.FC<GlassBadgeProps> = ({
  children,
  variant = 'cyan',
  size = 'md',
  pulse = false,
  className = '',
}) => {
  const styles = {
    cyan: 'bg-cyan-500/20 dark:bg-cyan-500/20 light:bg-sky-100 text-cyan-300 dark:text-cyan-300 light:text-[#00639b] border-cyan-400/50 dark:border-cyan-400/50 light:border-sky-300 shadow-[0_0_10px_rgba(34,211,238,0.2),inset_0_1px_1px_rgba(255,255,255,0.3)]',
    emerald:
      'bg-emerald-500/20 dark:bg-emerald-500/20 light:bg-emerald-100 text-emerald-300 dark:text-emerald-300 light:text-[#03624c] border-emerald-400/50 dark:border-emerald-400/50 light:border-emerald-300 shadow-[0_0_10px_rgba(16,185,129,0.2),inset_0_1px_1px_rgba(255,255,255,0.3)]',
    amber:
      'bg-amber-500/20 dark:bg-amber-500/20 light:bg-amber-100 text-amber-300 dark:text-amber-300 light:text-[#8a3b00] border-amber-400/50 dark:border-amber-400/50 light:border-amber-300 shadow-[0_0_10px_rgba(245,158,11,0.2),inset_0_1px_1px_rgba(255,255,255,0.3)]',
    rose: 'bg-rose-500/20 dark:bg-rose-500/20 light:bg-rose-100 text-rose-300 dark:text-rose-300 light:text-[#9e1030] border-rose-400/50 dark:border-rose-400/50 light:border-rose-300 shadow-[0_0_10px_rgba(244,63,94,0.2),inset_0_1px_1px_rgba(255,255,255,0.3)]',
    purple:
      'bg-purple-500/20 dark:bg-purple-500/20 light:bg-purple-100 text-purple-300 dark:text-purple-300 light:text-[#60259e] border-purple-400/50 dark:border-purple-400/50 light:border-purple-300 shadow-[0_0_10px_rgba(168,85,247,0.2),inset_0_1px_1px_rgba(255,255,255,0.3)]',
    slate:
      'bg-slate-800/70 dark:bg-slate-800/70 light:bg-slate-100 text-slate-300 dark:text-slate-300 light:text-[#163252] border-slate-600/50 dark:border-slate-600/50 light:border-slate-300 shadow-sm',
  };

  const sizes = {
    sm: 'px-2 py-0.5 text-[9px]',
    md: 'px-3 py-0.5 text-[11px]',
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-mono font-bold uppercase tracking-wider border backdrop-blur-xl relative overflow-hidden ${styles[variant]} ${sizes[size]} ${className}`}
    >
      <span className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-white/40 to-transparent pointer-events-none" />
      {pulse && (
        <span
          className={`w-1.5 h-1.5 rounded-full animate-pulse shrink-0 ${
            variant === 'emerald'
              ? 'bg-emerald-400 dark:bg-emerald-400 light:bg-emerald-600 shadow-[0_0_6px_#34d399]'
              : variant === 'amber'
              ? 'bg-amber-400 dark:bg-amber-400 light:bg-amber-600 shadow-[0_0_6px_#fbbf24]'
              : variant === 'rose'
              ? 'bg-rose-400 dark:bg-rose-400 light:bg-rose-600 shadow-[0_0_6px_#f43f5e]'
              : 'bg-cyan-400 dark:bg-cyan-400 light:bg-sky-600 shadow-[0_0_6px_#22d3ee]'
          }`}
        />
      )}
      <span className="relative z-10">{children}</span>
    </span>
  );
};

export interface KpiCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  liquid?: boolean;
}

export const KpiCard: React.FC<KpiCardProps> = ({
  children,
  className = '',
  liquid = true,
  ...props
}) => {
  return (
    <div
      className={`kpi-card ${className}`}
      {...props}
    >
      {/* Liquid Specular Top Sheen Edge */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-cyan-400/40 dark:via-cyan-400/40 light:via-sky-400/50 to-transparent pointer-events-none z-10" />

      {/* Convex Ambient Flare */}
      {liquid && (
        <div className="absolute top-0 left-0 w-36 h-24 bg-radial from-white/[0.08] dark:from-white/[0.08] light:from-white/[0.6] to-transparent pointer-events-none z-0" />
      )}

      <div className="relative z-10 flex flex-col justify-between flex-1">{children}</div>
    </div>
  );
};

