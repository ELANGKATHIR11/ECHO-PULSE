import { SonarFrame } from '../types';
import { fetchWithTimeout } from './api';

export const sonarApi = {
  /**
   * Uploads and parses a raw sonar file (XTF, GeoTIFF, PNG, NPY, SSS binary)
   */
  async uploadSonarFile(
    file: File,
    missionId: string
  ): Promise<{ fileId: string; pingsCount: number; frequencyKhz: number; detectionsCount?: number }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('missionId', missionId);

    const response = await fetch('/api/v1/sonar/upload', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload endpoint failed with status ${response.status}`);
    }
    return await response.json();
  },

  /**
   * Fetches acoustic frame by mission & ping index with real OpenCV metrics
   */
  async getFrame(missionId: string, pingIndex: number): Promise<SonarFrame> {
    return await fetchWithTimeout<SonarFrame>(`/sonar/frames/${missionId}?ping=${pingIndex}`);
  },
};
