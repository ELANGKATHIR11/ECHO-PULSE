import { ReportItem } from '../types';
import { fetchWithTimeout } from './api';

export const reportApi = {
  async getReports(missionId?: string): Promise<ReportItem[]> {
    const q = missionId ? `?mission_id=${missionId}` : '';
    return await fetchWithTimeout<ReportItem[]>(`/reports${q}`);
  },

  async generateReport(payload: {
    missionId: string;
    format: 'JSON' | 'CSV' | 'GeoJSON' | 'GeoPackage' | 'HTML/PDF';
    includeShadowMetrics: boolean;
    includeTelemetry: boolean;
  }): Promise<ReportItem> {
    return await fetchWithTimeout<ReportItem>('/reports/generate', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },
};
