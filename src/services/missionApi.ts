import { Mission } from '../types';
import { fetchWithTimeout } from './api';

export const missionApi = {
  async getMissions(): Promise<Mission[]> {
    return await fetchWithTimeout<Mission[]>('/missions');
  },

  async getMissionById(id: string): Promise<Mission | null> {
    return await fetchWithTimeout<Mission>(`/missions/${id}`);
  },

  async createMission(mission: Partial<Mission>): Promise<Mission> {
    return await fetchWithTimeout<Mission>('/missions', {
      method: 'POST',
      body: JSON.stringify(mission),
    });
  },

  async deleteMission(id: string): Promise<boolean> {
    await fetchWithTimeout(`/missions/${id}`, { method: 'DELETE' });
    return true;
  },
};
