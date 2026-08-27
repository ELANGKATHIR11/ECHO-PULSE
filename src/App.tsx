import React, { Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { CommandLayout } from './components/layout/CommandLayout';
import { HomePage } from './pages/HomePage';
import { DashboardPage } from './pages/DashboardPage';
import { DigitalTwinPage } from './pages/DigitalTwinPage';
import { DetectionsPage } from './pages/DetectionsPage';
import { DetectionDetailPage } from './pages/DetectionDetailPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { WebcamTrackerPage } from './pages/WebcamTrackerPage';
import { RawSonarUploadPage } from './pages/RawSonarUploadPage';
import { CommandCenterPage } from './pages/CommandCenterPage';
import { MpaDebrisMapPage } from './pages/MpaDebrisMapPage';
import { PostgresSpatialDataPage } from './pages/PostgresSpatialDataPage';
import { ErrorBoundary } from './components/common/ErrorBoundary';

const PageFallback = () => (
  <div className="flex items-center justify-center min-h-[60vh] text-cyan-400 font-mono text-xs">
    <div className="flex items-center gap-3 p-4 rounded-xl bg-[#020712]/70 border border-cyan-900/40">
      <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping" />
      <span>INITIALIZING WORKSTATION SURFACE...</span>
    </div>
  </div>
);

export default function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<CommandLayout />}>
              <Route
                path="/"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <HomePage />
                  </Suspense>
                }
              />
              <Route
                path="/command-center"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <CommandCenterPage />
                  </Suspense>
                }
              />
              <Route
                path="/dashboard"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <DashboardPage />
                  </Suspense>
                }
              />

              <Route
                path="/digital-twin"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <DigitalTwinPage />
                  </Suspense>
                }
              />
              <Route
                path="/webcam-tracker"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <WebcamTrackerPage />
                  </Suspense>
                }
              />
              <Route
                path="/detections"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <DetectionsPage />
                  </Suspense>
                }
              />
              <Route
                path="/detections/:id"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <DetectionDetailPage />
                  </Suspense>
                }
              />
              <Route
                path="/analytics"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <AnalyticsPage />
                  </Suspense>
                }
              />
              <Route
                path="/upload"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <RawSonarUploadPage />
                  </Suspense>
                }
              />
              <Route
                path="/postgres"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <PostgresSpatialDataPage />
                  </Suspense>
                }
              />
              <Route
                path="/mpa"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <MpaDebrisMapPage />
                  </Suspense>
                }
              />

              {/* Clean Aliases for streamlined architecture */}
              <Route path="/postgis" element={<Navigate to="/postgres" replace />} />
              <Route path="/database" element={<Navigate to="/postgres" replace />} />
              <Route path="/sql" element={<Navigate to="/postgres" replace />} />
              <Route path="/mpa-zones" element={<Navigate to="/mpa" replace />} />
              <Route path="/marine-protected-areas" element={<Navigate to="/mpa" replace />} />
              <Route path="/gio-tags" element={<Navigate to="/mpa" replace />} />
              <Route path="/geo-tags" element={<Navigate to="/mpa" replace />} />
              <Route path="/missions" element={<Navigate to="/dashboard" replace />} />
              <Route path="/missions/*" element={<Navigate to="/dashboard" replace />} />
              <Route path="/sonar" element={<Navigate to="/dashboard" replace />} />
              <Route path="/active-learning" element={<Navigate to="/upload" replace />} />
              <Route path="/vision" element={<Navigate to="/webcam-tracker" replace />} />
              <Route path="/camera" element={<Navigate to="/webcam-tracker" replace />} />
              <Route path="/live-vision" element={<Navigate to="/webcam-tracker" replace />} />
              <Route path="/models" element={<Navigate to="/analytics?tab=models" replace />} />
              <Route path="/datasets" element={<Navigate to="/analytics?tab=datasets" replace />} />
              <Route path="/system" element={<Navigate to="/analytics" replace />} />
              <Route path="/reports" element={<Navigate to="/analytics" replace />} />
              <Route path="/settings" element={<Navigate to="/analytics" replace />} />

              {/* Catch-all */}
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </ErrorBoundary>
  );
}
