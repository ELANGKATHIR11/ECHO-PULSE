import { DatasetInfo, ModelInfo, SystemTelemetry } from '../types';
import { fetchWithTimeout } from './api';

export const systemApi = {
  async getTelemetry(): Promise<SystemTelemetry> {
    return await fetchWithTimeout<SystemTelemetry>('/system/telemetry');
  },

  async getModels(): Promise<ModelInfo[]> {
    return await fetchWithTimeout<ModelInfo[]>('/models');
  },

  async getDatasets(): Promise<DatasetInfo[]> {
    return await fetchWithTimeout<DatasetInfo[]>('/datasets');
  },

  async triggerDatasetValidation(datasetId: string): Promise<{ success: boolean; message: string }> {
    return await fetchWithTimeout<{ success: boolean; message: string }>(`/datasets/${datasetId}/validate`, {
      method: 'POST',
    });
  },
};
