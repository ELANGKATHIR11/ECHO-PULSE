import { Detection } from '../types';
import { fetchWithTimeout } from './api';

export const detectionApi = {
  async getDetections(filters?: {
    missionId?: string;
    class?: string;
    minConfidence?: number;
    hasGeotag?: boolean;
  }): Promise<Detection[]> {
    const query = new URLSearchParams();
    if (filters?.missionId) query.set('mission_id', filters.missionId);
    if (filters?.class && filters.class !== 'ALL') query.set('class', filters.class);
    if (filters?.minConfidence) query.set('min_confidence', filters.minConfidence.toString());

    return await fetchWithTimeout<Detection[]>(`/detections?${query.toString()}`);
  },

  async getDetectionById(id: string): Promise<Detection | null> {
    return await fetchWithTimeout<Detection>(`/detections/${id}`);
  },

  async updateDetectionVerification(
    id: string,
    status: 'UNVERIFIED' | 'CONFIRMED' | 'FALSE_POSITIVE',
    notes?: string
  ): Promise<Detection> {
    return await fetchWithTimeout<Detection>(`/detections/${id}/verify`, {
      method: 'POST',
      body: JSON.stringify({ status, notes }),
    });
  },

  async createDetection(detection: Detection): Promise<Detection> {
    return await fetchWithTimeout<Detection>('/detections', {
      method: 'POST',
      body: JSON.stringify(detection),
    });
  },

  async deleteDetection(id: string): Promise<boolean> {
    await fetchWithTimeout(`/detections/${id}`, { method: 'DELETE' });
    return true;
  },
};
