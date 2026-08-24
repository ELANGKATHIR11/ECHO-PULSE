import { Detection, InferenceJobState } from '../types';
import { fetchWithTimeout } from './api';

export const inferenceApi = {
  /**
   * Start AI Sonar Inference Job
   */
  async startInference(payload: {
    missionId: string;
    modelId?: string;
    enableShadowFusion: boolean;
    enableAnomalyDetection: boolean;
    confidenceThreshold: number;
  }): Promise<{ jobId: string; status: string }> {
    return await fetchWithTimeout<{ jobId: string; status: string }>('/inference', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /**
   * Poll inference status
   */
  async getInferenceStatus(jobId: string, missionId?: string): Promise<InferenceJobState> {
    const q = missionId ? `?mission_id=${missionId}` : '';
    return await fetchWithTimeout<InferenceJobState>(`/inference/${jobId}${q}`);
  },
};
