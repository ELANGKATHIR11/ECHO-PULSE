# 8-Channel Physics-Guided Acoustic Tensor Specification

## Overview
EchoPulseNet extracts an 8-channel physics-guided tensor $\mathbf{X}_{\text{phys}} \in \mathbb{R}^{B \times 8 \times H \times W}$ directly from raw side-scan sonar (SSS) backscatter images and navigational/environmental metadata. This multi-modal tensor encodes fundamental underwater acoustics principles, transmission losses, grazing angles, and sound velocity profiles.

---

## Tensor Channel Definitions

| Channel | Identifier | Name | Formula / Derivation | Unit | Source | Normalization |
|---|---|---|---|---|---|---|
| **C1** | `raw_intensity` | Calibrated Acoustic Backscatter | $I(x,y) \in [0, 255]$ | Dimensionless intensity | Sonar sensor (XTF/JSF/Raster) | Linear $[0, 1]$ |
| **C2** | `substrate_reverb` | Low-Frequency Substrate Reverberation | $I_{\text{LF}} = \text{AvgPool}_{9\times9}(I)$ | Filtered intensity | Derived (2D Spatial Low-Pass) | $[0, 1]$ |
| **C3** | `highlight_contrast` | High-Frequency Specular Highlight Anomaly | $I_{\text{HF}} = \text{clamp}(I - I_{\text{LF}} + 0.5, 0, 1)$ | Contrast residual | Derived (High-pass residual) | $[0, 1]$ |
| **C4** | `scatter_gradient` | Local Acoustic Scattering Variance | $\sigma_{\text{local}} = \text{clamp}(\|I - \text{AvgPool}_{19\times19}(I)\| \cdot 3.2, 0, 1)$ | Biofouling/Roughness proxy | Derived (Bandpass variance) | $[0, 1]$ |
| **C5** | `cross_track_range` | Normalized Cross-Track Slant Range | $r_{\text{norm}}(x) = \frac{x}{W - 1}$ | $[0, 1]$ relative range | Swath Geometry | Linear $[0.05, 1.0]$ |
| **C6** | `transmission_loss` | Theoretical Oceanic Propagation Loss | $\text{TL}(r) = \frac{20\log_{10}(\max(r,1)) + \alpha(f) \cdot r}{60.0}$ | Normalized $\text{dB}$ | Calculated (Spherical + Absorption) | $[0, 1]$ clamped |
| **C7** | `sound_speed_field` | Deep Ocean Acoustic Sound Speed | $c(T,S,D) = 1448.96 + 4.591T - 0.05304T^2 + 1.34(S-35) + 0.0163D$ | $\text{m/s}$ | Mackenzie Equation ($T,S,D$ metadata/assumed) | Normalized $\frac{c}{1600.0}$ |
| **C8** | `grazing_angle` | Acoustic Grazing / Incident Angle | $\gamma(r, H) = \frac{\arctan(H / \max(r, 1))}{\pi / 2}$ | Normalized radians | Geometry ($H_{\text{alt}}$, $R_{\text{slant}}$) | $[0, 1]$ |

---

## Physical Formulations & Oceanographic Physics

### 1. Ainslie-McColm Oceanic Absorption $\alpha(f)$
Absorption coefficient in $\text{dB/km}$ for acoustic frequency $f$ in $\text{kHz}$:
$$\alpha(f) = \frac{0.106 f^2}{f^2 + 36.0} + 0.00049 f^2$$
For standard high-resolution $450\text{ kHz}$ side-scan sonar, the absorption loss is approximately $0.10\text{ dB/m}$.

### 2. Mackenzie (1981) Sound Velocity Equation
Sound velocity $c$ ($\text{m/s}$) as a function of Temperature $T$ ($^\circ\text{C}$), Salinity $S$ ($\text{ppt}$), and Depth $D$ ($\text{m}$):
$$c(T, S, D) = 1448.96 + 4.591T - 0.05304T^2 + 1.34(S - 35.0) + 0.0163D$$
- **Measured Data**: Extracted when CTD (Conductivity-Temperature-Depth) sensor packets are present in the sonar stream.
- **Supplied / Simulated Defaults**: Standard ocean surface baseline ($T=4^\circ\text{C}$, $S=35\text{ ppt}$, $D=32\text{ m}$, yielding $c \approx 1466\text{ m/s}$).

### 3. Slant-Range to Ground-Range Correction
$$R_{\text{ground}} = \sqrt{\max(0, R_{\text{slant}}^2 - H_{\text{altitude}}^2)}$$
Valid only when $R_{\text{slant}} \ge H_{\text{altitude}}$.

### 4. Acoustic Shadow Target Height Inversion
The target height above the seafloor $H_{\text{target}}$ is estimated from shadow length $L_{\text{shadow}}$, sensor altitude $H_{\text{sensor}}$, and slant range $R_{\text{slant}}$:
$$H_{\text{target}} = \frac{L_{\text{shadow}} \cdot H_{\text{sensor}}}{R_{\text{slant}} + L_{\text{shadow}}}$$
When no shadow is detectable, $H_{\text{target}}$ is strictly reported as `null`.
