/**
 * Web Audio API Acoustic Sonar Sound Synthesizer
 * Generates realistic submarine sonar chirps, ping reverberation, and target lock-on audio cues.
 */

class SonarAudioEngine {
  private ctx: AudioContext | null = null;
  private isMuted: boolean = false;

  private getContext(): AudioContext | null {
    if (!this.ctx && typeof window !== 'undefined') {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
    return this.ctx;
  }

  public setMuted(muted: boolean) {
    this.isMuted = muted;
  }

  public getMuted(): boolean {
    return this.isMuted;
  }

  /**
   * Classic Submarine High-Frequency Ping Chirp (Frequency Modulated Sweep with Long Acoustic Reverb)
   */
  public playSonarPing(frequency: number = 880, duration: number = 1.2) {
    if (this.isMuted) return;
    const ctx = this.getContext();
    if (!ctx) return;

    const now = ctx.currentTime;

    // Oscillator for Ping
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sine';
    // Downward frequency chirp characteristic of active sidescan sonar pulses
    osc.frequency.setValueAtTime(frequency * 1.5, now);
    osc.frequency.exponentialRampToValueAtTime(frequency, now + 0.1);
    osc.frequency.exponentialRampToValueAtTime(frequency * 0.9, now + duration);

    // Sharp attack, exponential decay for underwater acoustic reverberation
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(0.25, now + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

    // Sub-bass thump for hull transducer vibration
    const subOsc = ctx.createOscillator();
    const subGain = ctx.createGain();
    subOsc.type = 'triangle';
    subOsc.frequency.setValueAtTime(120, now);
    subOsc.frequency.exponentialRampToValueAtTime(40, now + 0.3);
    subGain.gain.setValueAtTime(0.2, now);
    subGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.3);

    osc.connect(gain);
    subOsc.connect(subGain);
    gain.connect(ctx.destination);
    subGain.connect(ctx.destination);

    osc.start(now);
    subOsc.start(now);
    osc.stop(now + duration);
    subOsc.stop(now + 0.3);
  }

  /**
   * Target Lock-On Acoustic Beep (Double-Chirp Alert)
   */
  public playTargetLock() {
    if (this.isMuted) return;
    const ctx = this.getContext();
    if (!ctx) return;

    const now = ctx.currentTime;
    [0, 0.09].forEach((offset) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'square';
      osc.frequency.setValueAtTime(1760, now + offset);
      gain.gain.setValueAtTime(0.12, now + offset);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.06);

      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now + offset);
      osc.stop(now + offset + 0.06);
    });
  }
}

export const sonarAudio = new SonarAudioEngine();
