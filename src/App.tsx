import React, { Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { CommandLayout } from './components/layout/CommandLayout';
import { HomePage } from './pages/HomePage';
import { DashboardPage } from './pages/DashboardPage';
import { DigitalTwinPage } from './pages/DigitalTwinPage';
import { MissionsPage } from './pages/MissionsPage';
import { MissionDetailPage } from './pages/MissionDetailPage';
import { SonarWorkstationPage } from './pages/SonarWorkstationPage';
import { DetectionsPage } from './pages/DetectionsPage';
import { DetectionDetailPage } from './pages/DetectionDetailPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { DemoPage } from './pages/DemoPage';
import { WebcamTrackerPage } from './pages/WebcamTrackerPage';
import { ActiveLearningStudio } from './pages/ActiveLearningStudio';
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
                path="/missions"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <MissionsPage />
                  </Suspense>
                }
              />
              <Route
                path="/missions/:id"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <MissionDetailPage />
                  </Suspense>
                }
              />
              <Route
                path="/sonar"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <SonarWorkstationPage />
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
                path="/active-learning"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <ActiveLearningStudio />
                  </Suspense>
                }
              />
              <Route
                path="/demo"
                element={
                  <Suspense fallback={<PageFallback />}>
                    <DemoPage />
                  </Suspense>
                }
              />

              {/* Clean Aliases for streamlined architecture */}
              <Route path="/vision" element={<Navigate to="/webcam-tracker" replace />} />
              <Route path="/camera" element={<Navigate to="/webcam-tracker" replace />} />
              <Route path="/live-vision" element={<Navigate to="/webcam-tracker" replace />} />
              <Route path="/models" element={<Navigate to="/analytics?tab=models" replace />} />
              <Route path="/datasets" element={<Navigate to="/analytics?tab=datasets" replace />} />
              <Route path="/system" element={<Navigate to="/analytics?tab=system" replace />} />
              <Route path="/reports" element={<Navigate to="/missions" replace />} />
              <Route path="/settings" element={<Navigate to="/sonar" replace />} />

              {/* Catch-all */}
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </ErrorBoundary>
  );
}
