import { ColorPalette, Detection, SonarFrame, SonarViewerSettings } from '../types';

/**
 * Generates false color palettes for acoustic sonar rendering.
 */
export function getPaletteColor(intensity: number, palette: ColorPalette): [number, number, number] {
  // intensity is 0 to 255
  const norm = Math.max(0, Math.min(1, intensity / 255));

  switch (palette) {
    case 'copper': {
      // Classic side-scan bronze/copper: dark brown -> copper gold -> bright highlight
      const r = Math.min(255, Math.floor(norm * 255 * 1.15));
      const g = Math.min(255, Math.floor(Math.pow(norm, 1.25) * 195));
      const b = Math.min(255, Math.floor(Math.pow(norm, 1.6) * 125));
      return [r, g, b];
    }
    case 'amber_sonar': {
      // Golden acoustic amber
      const r = Math.min(255, Math.floor(Math.pow(norm, 0.9) * 255));
      const g = Math.min(255, Math.floor(Math.pow(norm, 1.3) * 180));
      const b = Math.min(255, Math.floor(Math.pow(norm, 2.5) * 45));
      return [r, g, b];
    }
    case 'oceanic_blue': {
      // Deep cyan/abyssal oceanic blue
      const r = Math.min(255, Math.floor(Math.pow(norm, 2.2) * 80));
      const g = Math.min(255, Math.floor(Math.pow(norm, 1.1) * 220));
      const b = Math.min(255, Math.floor(Math.pow(norm, 0.85) * 255));
      return [r, g, b];
    }
    case 'emerald': {
      // Military green phosphor / night vision sonar
      const r = Math.min(255, Math.floor(Math.pow(norm, 2.5) * 60));
      const g = Math.min(255, Math.floor(Math.pow(norm, 0.9) * 255));
      const b = Math.min(255, Math.floor(Math.pow(norm, 1.8) * 110));
      return [r, g, b];
    }
    case 'thermal': {
      // Scientific thermal jet (Blue -> Cyan -> Yellow -> Red -> White)
      if (norm < 0.25) {
        return [0, Math.floor(norm * 4 * 255), 255];
      } else if (norm < 0.5) {
        return [0, 255, Math.floor((1 - (norm - 0.25) * 4) * 255)];
      } else if (norm < 0.75) {
        return [Math.floor((norm - 0.5) * 4 * 255), 255, 0];
      } else {
        return [255, Math.floor((1 - (norm - 0.75) * 4) * 255), Math.floor((norm - 0.75) * 4 * 255)];
      }
    }
    case 'grayscale':
    default: {
      const v = Math.floor(norm * 255);
      return [v, v, v];
    }
  }
}

/**
 * Generates an ultra-realistic synthetic sonar waterfall image onto an offscreen HTMLCanvasElement
 * containing:
 * - Water column (nadir blind zone in middle)
 * - Seabed backscatter with sand ripples & acoustic speckle
 * - Specular acoustic highlights from seabed targets
 * - Trailing acoustic acoustic shadow zone (no acoustic return)
 */
export function generateSyntheticSonarCanvas(
  width: number = 800,
  height: number = 500,
  detections: Detection[] = [],
  palette: ColorPalette = 'copper',
  isProcessed: boolean = false
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  if (!ctx) return canvas;

  const imgData = ctx.createImageData(width, height);
  const data = imgData.data;

  const nadirWidth = Math.floor(width * 0.08); // 8% center nadir
  const centerX = width / 2;

  // Generate seabed texture
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const distFromCenter = Math.abs(x - centerX);
      let intensity = 0;

      if (distFromCenter < nadirWidth) {
        // Water column blind zone - very dark with faint water column reverberations
        const falloff = distFromCenter / nadirWidth;
        intensity = Math.floor(falloff * 20 + (Math.sin(y * 0.05 + x * 0.1) * 5 + 5));
      } else {
        // Seabed grazing acoustic backscatter (Lambertian model + ripple noise)
        const rangeRatio = (distFromCenter - nadirWidth) / (centerX - nadirWidth);
        const grazingAngleAtten = Math.max(0.2, 1 - Math.pow(rangeRatio, 1.4) * 0.55);
        
        // Pseudo-random acoustic speckle and seabed sand ripples
        const ripple = Math.sin(x * 0.08 + Math.cos(y * 0.02) * 4) * 18;
        const microNoise = (Math.sin(x * 12.3 + y * 7.7) * 43758.5453) % 1;
        const speckle = (Math.abs(microNoise) - 0.5) * 35;
        
        const baseSeabed = (110 + ripple + speckle) * grazingAngleAtten;
        intensity = Math.max(10, Math.min(240, baseSeabed));
      }

      // Check if inside target highlights or shadows
      for (const det of detections) {
        const bx = det.bbox.x * width;
        const by = det.bbox.y * height;
        const bw = det.bbox.width * width;
        const bh = det.bbox.height * height;

        // Target highlight
        if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
          const dx = (x - (bx + bw / 2)) / (bw / 2);
          const dy = (y - (by + bh / 2)) / (bh / 2);
          if (dx * dx + dy * dy <= 1.2) {
            // Specular hard reflection
            intensity = Math.min(255, intensity + 120 + Math.random() * 30);
          }
        }

        // Acoustic shadow (stretches outward from nadir behind target)
        if (det.acousticShadow) {
          const isRight = bx > centerX;
          const shadowX = isRight ? bx + bw : bx - det.acousticShadow.lengthMeters * 10;
          const shadowW = det.acousticShadow.lengthMeters * 12;
          
          if (
            (isRight && x >= bx + bw * 0.8 && x <= bx + bw + shadowW) ||
            (!isRight && x <= bx + bw * 0.2 && x >= bx - shadowW)
          ) {
            if (y >= by && y <= by + bh * 1.1) {
              // Deep acoustic shadow (sound blocked)
              intensity = Math.max(2, intensity * 0.08 - 10);
            }
          }
        }
      }

      // If processed, apply TVG (Time Varied Gain) normalization & speckle reduction
      if (isProcessed) {
        intensity = Math.min(255, Math.pow(intensity / 255, 0.9) * 255 * 1.1);
      }

      const [r, g, b] = getPaletteColor(intensity, palette);
      const idx = (y * width + x) * 4;
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);

  // If processed, render subtle OpenCV contour annotations if wanted
  if (isProcessed) {
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    // draw center nadir line
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  return canvas;
}

/**
 * Calculates real-time image histogram (256 bins)
 */
export function computeImageHistogram(ctx: CanvasRenderingContext2D, width: number, height: number): number[] {
  const imgData = ctx.getImageData(0, 0, width, height);
  const data = imgData.data;
  const hist = new Array(256).fill(0);

  for (let i = 0; i < data.length; i += 4) {
    // Luminance
    const lum = Math.floor(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    hist[lum]++;
  }

  // Normalize to 0-100 max
  const maxVal = Math.max(...hist, 1);
  return hist.map((v) => (v / maxVal) * 100);
}

/**
 * Apply real-time client adjustments (Brightness, Contrast, Gamma, Threshold)
 */
export function applyViewerFilters(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  settings: SonarViewerSettings
) {
  if (
    settings.brightness === 0 &&
    settings.contrast === 0 &&
    settings.gamma === 1.0 &&
    !settings.thresholdPreview &&
    !settings.invert
  ) {
    return; // No adjustments
  }

  const imgData = ctx.getImageData(0, 0, width, height);
  const data = imgData.data;

  const contrastFactor = (259 * (settings.contrast + 255)) / (255 * (259 - settings.contrast));
  const gammaExp = 1 / Math.max(0.1, settings.gamma);

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i];
    let g = data[i + 1];
    let b = data[i + 2];

    // Grayscale luminance
    let gray = 0.299 * r + 0.587 * g + 0.114 * b;

    // Brightness
    gray += settings.brightness * 1.2;

    // Contrast
    gray = contrastFactor * (gray - 128) + 128;

    // Gamma
    gray = 255 * Math.pow(Math.max(0, gray) / 255, gammaExp);

    // Invert
    if (settings.invert) {
      gray = 255 - gray;
    }

    // Threshold Preview (Binary Mask mode)
    if (settings.thresholdPreview) {
      gray = gray >= settings.thresholdLevel ? 255 : 0;
    }

    gray = Math.max(0, Math.min(255, gray));

    // Remap through current palette if modified
    const [pr, pg, pb] = getPaletteColor(gray, settings.palette);
    data[i] = pr;
    data[i + 1] = pg;
    data[i + 2] = pb;
  }

  ctx.putImageData(imgData, 0, 0);
}

/**
 * Formats coordinates into standard maritime navigation degrees minutes seconds
 */
export function formatDMS(val: number, isLat: boolean): string {
  const dir = isLat ? (val >= 0 ? 'N' : 'S') : val >= 0 ? 'E' : 'W';
  const abs = Math.abs(val);
  const deg = Math.floor(abs);
  const minFloat = (abs - deg) * 60;
  const min = Math.floor(minFloat);
  const sec = ((minFloat - min) * 60).toFixed(2);
  return `${deg}° ${min}' ${sec}" ${dir}`;
}
