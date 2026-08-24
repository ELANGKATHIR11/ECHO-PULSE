import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error caught by ErrorBoundary:', error, errorInfo);
  }

  private handleReload = () => {
    window.location.reload();
  };

  private handleGoHome = () => {
    window.location.href = '/';
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen w-full bg-[#020712] text-slate-100 flex items-center justify-center p-6 font-mono">
          <div className="max-w-lg w-full p-6 rounded-2xl bg-[#050e1f]/90 border border-cyan-500/30 backdrop-blur-xl shadow-2xl space-y-4">
            <div className="flex items-center gap-3">
              <div className="p-3 rounded-xl bg-rose-500/20 border border-rose-500/40 text-rose-400">
                <AlertTriangle className="w-6 h-6 animate-pulse" />
              </div>
              <div>
                <h2 className="text-base font-bold text-white uppercase tracking-wider">
                  {this.props.fallbackTitle || 'Workstation Exception Handled'}
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  The application encountered a runtime anomaly and gracefully recovered.
                </p>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-[#020610] border border-cyan-900/40 text-xs text-rose-300 font-mono overflow-auto max-h-32">
              {this.state.error?.message || 'Unknown initialization error'}
            </div>

            <div className="flex items-center gap-3 pt-2">
              <button
                onClick={this.handleReload}
                className="flex-1 py-2 px-4 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-[#020712] font-bold text-xs flex items-center justify-center gap-2 transition-all shadow-[0_0_15px_rgba(34,211,238,0.3)]"
              >
                <RefreshCw className="w-4 h-4" />
                RELOAD APPLICATION
              </button>
              <button
                onClick={this.handleGoHome}
                className="py-2 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold text-xs flex items-center justify-center gap-2 border border-slate-700 transition-all"
              >
                <Home className="w-4 h-4" />
                DASHBOARD
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
