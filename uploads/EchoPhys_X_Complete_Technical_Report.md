
# EchoPhys-X
## Complete Technical, Mathematical, Dataset and Deep-Learning Specification

**Project:** EchoPhys-X — Physics-Constrained, Multi-Frequency/Biofouling-Aware Side-Scan Sonar Intelligence  
**Current dataset-specific implementation:** EchoPhys-X-SSS640  
**Document status:** Research/engineering specification; not a patentability opinion  
**Date:** 25 August 2026

---

## 1. Executive Summary

EchoPhys-X is a proposed side-scan-sonar (SSS) perception framework designed to detect, localize, classify and assess anthropogenic marine debris under difficult acoustic conditions.

The full research concept combines:

1. sonar/range-aware preprocessing,
2. multi-scale deep feature extraction,
3. directional linear-cost contextual mixing,
4. target–shadow reasoning,
5. biological/biofouling-aware interpretation,
6. natural-rock/coral mimic rejection,
7. uncertainty-aware decision making,
8. physics-consistency validation,
9. optional paired-frequency fusion,
10. abstention when evidence is insufficient.

The original HydroMamba-V2 scaffold supplied with the project used a dual-frequency “bio stripper”, a state-space module, row-wise attention, and a single dense detection tensor plus a binary mask. The audit identified that the row-wise attention still forms `[B,H,W,W]`, so it is quadratic in range width rather than truly linear, and that the so-called Biot-Stoll inversion was only a learned convolutional subtraction rather than an explicit acoustic inversion.

The current dataset-optimized implementation corrects the practical input mismatch. It is designed for the uploaded 640×640 grayscale SSS dataset with four classes and uses a five-channel representation, an 80×80 / 40×40 / 20×20 feature pyramid, an anchor-free objectness/class/LTRB head, and lightweight depthwise-separable blocks.

This dataset-specific model is called **EchoPhys-X-SSS640**.

Important scientific boundary:

> EchoPhys-X-SSS640 currently operates on a single grayscale SSS dataset. Its LF/HF channels are deterministic image-derived proxies, not measured dual-frequency sonar channels.

The full physics-rich EchoPhys-X formulation should only claim measured LF/HF fusion when paired LF/HF acquisitions and environmental metadata are available.

---

# 2. Source Basis

## 2.1 Original supplied architecture

The supplied PDF describes HydroMamba-V2 as a dual-frequency network ingesting low-frequency 300 kHz and high-frequency 900 kHz sonar. It describes a PoroElasticBioStripper, VectorizedSpatialSSM, DirectionalRangeAttention and multi-task output heads. The original code produces a 9-channel dense box tensor and a binary mask. [Source: uploaded HydroMamba-V2 audit PDF.]

The original implementation explicitly used learned LF/HF convolutional branches followed by:

`clean_substrate = core_penetration - surface_scatter * biot_alpha`

with `biot_alpha` initialized as 0.45. This is a learned feature subtraction and should not be described as a literal Biot-Stoll acoustic inversion.

The original attention implementation creates `[B,H,W,W]` attention matrices. This is not O(HW) linear complexity.

## 2.2 Uploaded dataset

The uploaded ZIP was inspected directly.

Observed structure:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── sample_submission.csv
```

Dataset statistics observed during the sandbox analysis:

- training images: 402
- validation images: 110
- training objects: 677
- validation objects: 172
- classes: 4 (`0, 1, 2, 3`)
- image size: 640 × 640 grayscale
- labels: YOLO-style bounding boxes

Combined labeled objects:

\[
677+172=849
\]

The object-area analysis found:

- 252 objects with normalized box area below 0.01
- 341 objects with normalized box area above 0.10

This strongly motivates a multi-scale architecture.

The dataset does not provide paired LF/HF channels, biological-cover masks, shadow masks, bathymetry, or CTD/environmental measurements.

---

# 3. Design Objectives

EchoPhys-X is designed to satisfy the following:

### Detection

- small target detection
- medium target detection
- large object detection
- weak-contrast target detection
- multiple targets per sonar frame

### Interpretation

- natural seabed
- rock/coral-like formations
- biological/vegetation regions
- visible debris
- biofouled debris
- partially buried debris
- unknown/insufficient-evidence anomalies

### Robustness

- range-dependent intensity variation
- seabed variation
- acoustic shadows
- speckle/noise
- scale variation
- acquisition variation

### Deployment

- lightweight parameter count
- no quadratic global attention
- CPU/GPU compatibility
- offline/local inference potential

---

# 4. EchoPhys-X Conceptual Architecture

```text
                 SIDE-SCAN SONAR
                       |
             +---------+---------+
             |                   |
          LF channel          HF channel
             |                   |
             +---------+---------+
                       |
             Acoustic / Geometry
              parameter estimation
                       |
          +------------+-------------+
          |                          |
      Seabed field             Biofouling field
          |                          |
          +------------+-------------+
                       |
              Frequency fusion
                       |
             Multi-scale backbone
                       |
        +--------------+--------------+
        |              |              |
       P3             P4             P5
   small targets   medium targets  large targets
        |              |              |
        +--------------+--------------+
                       |
                target features
                       |
       +---------------+---------------+
       |               |               |
    detection        biology         shadow
       |               |               |
       +---------------+---------------+
                       |
              target-shadow graph
                       |
        physics/counterfactual check
                       |
           uncertainty calibration
                       |
              final decision
```

The dataset-specific SSS640 implementation currently instantiates the practical detector branch of this architecture.

---

# 5. Input Representation

## 5.1 Full EchoPhys-X input

The full system defines a pixel/sample observation vector:

\[
\mathbf{x}_p =
[
I^{LF}_p,
I^{HF}_p,
r_p,
\gamma_p,
f_p,
h_s,
c_p,
\alpha_p,
q_p,
T_p,
S_p,
P_p
]
\]

where:

- \(I^{LF}\): low-frequency sonar intensity
- \(I^{HF}\): high-frequency sonar intensity
- \(r\): slant range
- \(\gamma\): grazing/incidence angle
- \(f\): sonar operating frequency
- \(h_s\): sonar altitude
- \(c\): sound speed
- \(\alpha\): attenuation
- \(q\): signal-quality/SNR feature
- \(T\): temperature
- \(S\): salinity
- \(P\): pressure/depth

## 5.2 Dataset-compatible input

For the uploaded dataset, EchoPhys-X-SSS640 creates five deterministic image channels:

\[
X =
[
X_{raw},
X_{LFproxy},
X_{HFproxy},
X_{local},
X_{range}
]
\]

### Channel 0: raw calibrated intensity

\[
X_{raw}=I
\]

### Channel 1: low-frequency proxy

\[
X_{LFproxy} = G_{\sigma_1}*I
\]

where \(G_{\sigma_1}\) is a Gaussian blur.

### Channel 2: high-frequency residual proxy

\[
X_{HFproxy} =
clip(I-X_{LFproxy}+0.5,0,1)
\]

### Channel 3: local contrast / texture

\[
X_{local}
=
clip(
|I-G_{\sigma_2}*I|\cdot 3,
0,1
)
\]

### Channel 4: normalized range coordinate

\[
X_{range}(u,v)
=
\frac{v}{W-1}
\]

This is a geometry proxy, not a measured slant-range field.

The actual source code implements these five channels directly. fileciteturn3file0L63-L79

---

# 6. Acoustic Physics Layer

## 6.1 Basic range relation

For a measured two-way travel time \(t\):

\[
\boxed{
r = \frac{ct}{2}
}
\]

The dataset-compatible model does not claim to know \(c\) or \(t\); these become available when sonar metadata are present.

## 6.2 Sound speed

A physical model can be written as:

\[
c = f_c(T,S,P)
\]

and optionally a bounded learned correction:

\[
\hat c = c_{physical} + \Delta c_\theta
\]

with \(\Delta c_\theta\) constrained to a realistic interval.

## 6.3 Grazing angle

With sonar elevation \(z_s\), seabed elevation \(z_b\), and horizontal ground range \(g\):

\[
r = \sqrt{g^2+(z_s-z_b)^2}
\]

\[
\gamma =
\tan^{-1}
\left(
\frac{z_s-z_b}{g}
\right)
\]

When bathymetry is unavailable, \(\gamma\) should be treated as an estimated field with uncertainty.

## 6.4 Propagation loss

A practical sonar propagation approximation uses:

\[
TL(r,f)=20\log_{10}(r)+\alpha(f)r
\]

where \(\alpha(f)\) is frequency-dependent absorption/attenuation.

The sonar-equation family is commonly written as:

\[
EL = SL - TL_T - TL_R + TS
\]

where:

- \(EL\): received echo level
- \(SL\): source level
- \(TL_T\): transmit-path loss
- \(TL_R\): receive-path loss
- \(TS\): target strength

A SSS simulator source describes this form and the frequency-dependent absorption term. citeturn146010search8turn146010search10

---

# 7. Physics-Guided Scene Decomposition

The full research model uses latent fields:

\[
\mathcal{Z}
=
\{
S_b,D,B,H,S_h,N
\}
\]

where:

- \(S_b\): seabed response
- \(D\): anthropogenic debris
- \(B\): biological/biofouling response
- \(H\): target geometry/height
- \(S_h\): acoustic-shadow field
- \(N\): noise/artifact field

A differentiable reconstruction is:

\[
\hat I =
\mathcal R(
S_b,D,B,H,S_h,N;
r,\gamma,f,c,\alpha
)
\]

and a practical log-response model can be expressed as:

\[
y=\log(I+\epsilon)
\]

\[
\hat y =
\beta_0+
\beta_bS_b+
\beta_dD+
\beta_BB+
\beta_hH+
\beta_sS_h
\]

\[
y=\hat y+\epsilon_y
\]

This is a research formulation. The current dataset does not contain the labels needed to identify each latent component directly.

Recent research independently supports physics-guided decomposition of SSS into interpretable components such as seabed reflectivity, terrain/elevation and propagation loss. citeturn146010academia30

---

# 8. Biofouling and Biological Cover

A key EchoPhys-X objective is to prevent algae, moss, seagrass and other underwater growth from causing debris to be classified as natural rock/coral.

Define:

\[
B_p=
[
B_{moss},
B_{algae},
B_{seagrass},
B_{kelp},
B_{biofilm}
]
\]

and aggregate cover:

\[
C_{bio,p}
=
1-\prod_k(1-B_{k,p})
\]

The hidden-object probability becomes:

\[
P(D\mid I,B,S,G)
\]

rather than merely:

\[
P(D\mid I)
\]

This distinction is central to the scientific concept, but cannot be validated using the uploaded ZIP because biological-cover masks are absent.

---

# 9. Counterfactual De-Biofouling

EchoPhys-X proposes a latent counterfactual:

### Actual scene

\[
Z_{actual}
=
f(I,D,B,S_b,S_h,G)
\]

### Counterfactual scene

\[
Z_{clean}
=
f(I,D,0,S_b,S_h,G)
\]

The biological contribution is:

\[
\Delta_B = Z_{clean}-Z_{actual}
\]

and the debris representation is:

\[
F_D=
F_{actual}+
\lambda_B\Delta_B
\]

This is proposed as a research mechanism, not a demonstrated physical inversion.

---

# 10. Target–Shadow Geometry

For a candidate target with height \(H_t\), one simplified geometric relation can be written as:

\[
\boxed{
L_s \approx \frac{H_t}{\tan\gamma}
}
\]

The exact relation depends on sonar configuration, geometry and coordinate conventions, so this should be treated as a model component rather than a universal equation.

A more flexible EchoPhys-X expression is:

\[
L_{pred}
=
\frac{\hat H_t}
{\tan(\gamma+\Delta\gamma)}
+\Delta L_{bio}
\]

Then define the shadow-consistency error:

\[
\boxed{
E_{shadow}
=
\frac{
|L_{obs}-L_{pred}|
}{
L_{obs}+\epsilon
}
}
\]

A physics-informed SSS detector published in 2026 similarly uses target-height/shadow-length consistency as a geometric constraint. citeturn146010search1

Therefore, shadow geometry is useful but should not be claimed as EchoPhys-X's sole novelty.

---

# 11. Target–Biology–Shadow Graph

For target candidates \(V=\{v_1,\dots,v_n\}\), construct edges:

\[
e_{ij}
=
f(
d_{ij},
\Delta \theta_{ij},
\Delta r_{ij},
B_{ij},
S_{ij}
)
\]

Then:

\[
F_G = GNN(V,E)
\]

The graph represents relationships such as:

```text
target ↔ shadow
target ↔ vegetation
target ↔ seabed
```

This is intended to reduce isolated appearance-based decisions.

---

# 12. Natural-Rock / Coral Mimic Rejection

Use a dedicated decision variable:

\[
P_{natural}
=
P(\text{natural formation})
\]

and:

\[
P_{anthro}
=
P(\text{anthropogenic object})
\]

with a final evidence score:

\[
C=
w_dC_d+
w_sC_s+
w_gC_g+
w_fC_f+
w_pC_p+
w_qC_q
\]

where the terms represent:

- DL detection evidence
- shadow consistency
- geometry consistency
- frequency consistency
- physics consistency
- data quality

---

# 13. Multi-Scale Deep-Learning Architecture

The uploaded dataset makes multi-scale processing particularly important because a substantial number of objects are extremely small while many other objects occupy a large fraction of the image.

EchoPhys-X-SSS640 therefore uses:

\[
P3=80\times80
\]

\[
P4=40\times40
\]

\[
P5=20\times20
\]

for 640×640 inputs.

These correspond to approximately:

- stride 8
- stride 16
- stride 32

The source implementation explicitly defines these stages and comments P3 as the small-object level. fileciteturn3file0L185-L195

---

# 14. Backbone

Current dataset-specific backbone:

```text
5 channels
   ↓
32 channels, stride 2
   ↓
64 channels, stride 4
   ↓
96 channels, stride 8  → P3
   ↓
160 channels, stride 16 → P4
   ↓
224 channels, stride 32 → P5
```

The actual implementation uses depthwise-separable residual blocks and lightweight directional mixers. fileciteturn3file0L140-L195

---

# 15. Directional Mixer

The current dataset version deliberately avoids global quadratic attention.

It uses:

\[
R = DWConv_{1\times7}(X)
\]

\[
C = DWConv_{7\times1}(X)
\]

\[
G=\sigma(Conv_{1\times1}(X))
\]

and:

\[
\boxed{
Y=X+Conv_{1\times1}
\left(
G\odot R+(1-G)\odot C
\right)
}
\]

This is linear in the number of pixels for fixed kernel sizes:

\[
O(HWC)
\]

rather than the original row-attention operation:

\[
O(HW^2C)
\]

The supplied original model created an explicit `[B,H,W,W]` attention tensor, so the earlier claim of fully linear range attention was incorrect. fileciteturn3file2L502-L534

---

# 16. Feature Pyramid Network

The P3/P4/P5 feature maps are projected to 128 channels.

Top-down fusion:

\[
Q_5 = Conv(P_5)
\]

\[
Q_4 =
Refine(
[
Conv(P_4),
Up(Q_5)
]
)
\]

\[
Q_3 =
Refine(
[
Conv(P_3),
Up(Q_4)
]
)
\]

The uploaded implementation uses nearest-neighbor upsampling and depthwise-separable refinement. fileciteturn3file0L198-L210

---

# 17. Anchor-Free Detection Head

Each scale outputs:

1. objectness
2. class logits
3. four box distances

So:

\[
O\in\mathbb R^{1\times H\times W}
\]

\[
C\in\mathbb R^{K\times H\times W}
\]

\[
B\in\mathbb R^{4\times H\times W}
\]

and:

\[
B=softplus(B_{raw})
\]

ensures positive LTRB distances.

The current source implements exactly this structure. fileciteturn3file0L213-L234

---

# 18. Bounding-Box Parameterization

For cell \((g_x,g_y)\), predict:

\[
(l,t,r,b)
\]

Then:

\[
x_1=x_c-l
\]

\[
y_1=y_c-t
\]

\[
x_2=x_c+r
\]

\[
y_2=y_c+b
\]

For a normalized ground-truth box:

\[
(cx,cy,w,h)
\]

the target distances in feature-cell coordinates are:

\[
l=f_x-\frac{wW}{2}
\]

\[
t=f_y-\frac{hH}{2}
\]

\[
r=wW-f_x
\]

\[
b=hH-f_y
\]

where:

\[
f_x=c_xW-g_x
\]

\[
f_y=c_yH-g_y
\]

The dataset-specific implementation performs this assignment. fileciteturn3file0L237-L261

---

# 19. Scale Assignment

EchoPhys-X-SSS640 assigns objects to detection levels using normalized box area.

Current rule:

### P3 / stride 8

```text
area <= 0.08
```

### P4 / stride 16

```text
0.01 <= area <= 0.20
```

### P5 / stride 32

```text
area >= 0.04
```

These ranges overlap intentionally so that medium-scale objects can receive appropriate context. The current code implements these thresholds directly. fileciteturn3file0L247-L259

For production training, these thresholds should be tuned using validation data rather than treated as universally optimal.

---

# 20. Focal Binary Classification Loss

For logits \(z\), target \(y\), and:

\[
p=\sigma(z)
\]

define:

\[
p_t=
py+(1-p)(1-y)
\]

and:

\[
\alpha_t=
\alpha y+(1-\alpha)(1-y)
\]

then:

\[
\boxed{
L_{focal}
=
\alpha_t(1-p_t)^\gamma
BCE(z,y)
}
\]

Current implementation:

\[
\gamma=2,\qquad \alpha=0.25
\]

The source implements this focal-BCE formulation. fileciteturn3file0L264-L267

---

# 21. Box Regression Loss

Current baseline:

\[
L_{box}
=
SmoothL1(
B_{pred},B_{target}
)
\]

and the total per-level contribution is:

\[
L_{level}
=
L_{obj}+L_{cls}+2L_{box}
\]

The source implements the factor of 2 on the box term. fileciteturn3file0L270-L285

For a stronger production version, CIoU/EIoU or another geometry-aware loss should be evaluated experimentally rather than assumed to improve performance.

---

# 22. Total Dataset-Specific Detection Loss

\[
\boxed{
L_{det}
=
\sum_{s\in\{P3,P4,P5\}}
\left(
L_{obj}^{(s)}
+
L_{cls}^{(s)}
+
2L_{box}^{(s)}
\right)
}
\]

This is the current trainable objective implemented by EchoPhys-X-SSS640.

---

# 23. Full Research EchoPhys-X Loss

When complete annotations and physical metadata become available:

\[
\boxed{
L_{total}=
\lambda_D L_D+
\lambda_M L_M+
\lambda_B L_B+
\lambda_S L_S+
\lambda_G L_G+
\lambda_F L_F+
\lambda_P L_P+
\lambda_C L_C+
\lambda_U L_U+
\lambda_R L_R
}
\]

where:

- \(L_D\): detection
- \(L_M\): debris segmentation
- \(L_B\): biological-cover segmentation
- \(L_S\): shadow segmentation
- \(L_G\): geometry
- \(L_F\): frequency consistency
- \(L_P\): acoustic reconstruction/physics consistency
- \(L_C\): graph consistency
- \(L_U\): uncertainty calibration
- \(L_R\): counterfactual/de-biofouling consistency

The current ZIP does not support all of these terms.

---

# 24. Uncertainty Model

For a regression quantity \(y\), predict mean \(\mu\) and log variance \(s\):

\[
\boxed{
L_{het}
=
\frac12e^{-s}(y-\mu)^2+\frac12s
}
\]

For final object confidence:

\[
\boxed{
C_{final}
=
C_{DL}
\cdot
C_{physics}
\cdot
C_{shadow}
\cdot
C_{frequency}
\cdot
C_{quality}
\cdot
(1-U)
}
\]

When:

\[
C_{final}<\tau_{reject}
\]

the model should abstain:

```text
UNKNOWN / INSUFFICIENT EVIDENCE
```

instead of forcing a class.

---

# 25. Biofouled-Debris Decision

A key research decision rule is:

\[
P_{biofouled}
=
P(
D\land B
)
\]

rather than interpreting vegetation as the inverse of debris.

A practical decision vector is:

\[
R=
[
P_D,
P_B,
P_N,
P_{unknown},
C_{shadow},
C_{physics},
U
]
\]

and:

```text
high P_D + high P_B
    -> biofouled debris

high P_B + low P_D
    -> biological formation

high P_N + low P_D
    -> natural substrate

low evidence / high uncertainty
    -> unknown
```

This is a proposed research design; the uploaded dataset does not contain the biology labels needed to train it.

---

# 26. Dataset Augmentation

The current source deliberately uses sonar-safe augmentation:

- intensity gain: 0.92–1.08
- intensity bias: -0.04 to 0.04
- Gaussian noise around 0.018 with probability 0.30
- local attenuation/dropout with probability 0.20

Vertical flipping is disabled because it can invert range-related orientation. Strong rotation is also avoided.

These settings are present in the actual source. fileciteturn3file0L113-L132

---

# 27. Dataset Imbalance

Training objects:

\[
[71,169,102,335]
\]

per the earlier dataset inspection.

Combined train+validation counts:

\[
[88,207,139,415]
\]

Thus class 3 is substantially more represented than class 0.

The recommended production sampler is therefore:

\[
w_c=
\frac{1}
{n_c^\beta}
\]

with:

\[
0<\beta<1
\]

rather than simple inverse-frequency weighting, to avoid overcorrecting the minority class.

---

# 28. Small-Object Strategy

Because 252 labeled objects are below 1% normalized box area, EchoPhys-X-SSS640 prioritizes P3.

Recommended additional strategy:

\[
w_{small}>w_{medium}>w_{large}
\]

for the objectness loss.

A practical smooth weight:

\[
w_{size}
=
clip
\left(
\sqrt{
\frac{a_{ref}}
{a+\epsilon}
},
w_{min},
w_{max}
\right)
\]

where \(a\) is normalized object area.

This should be validated experimentally rather than hard-coded without an ablation.

---

# 29. Hard Negatives

The system should explicitly learn from:

- natural rock
- coral-like formations
- seabed texture
- vegetation clusters
- shadow-only regions
- sonar artifacts
- sediment mounds

A hard-negative loss can be added:

\[
L_{HN}
=
\lambda_{HN}
L_{obj}^{hard}
\]

to reduce false positives.

This is especially important for the final biofouling-aware system.

---

# 30. Validation Protocol

Do not judge the system with only one global mAP.

Report:

\[
mAP_{50}
\]

\[
mAP_{50:95}
\]

\[
Precision
\]

\[
Recall
\]

\[
F1
\]

and class-specific values.

Also report:

\[
mAP_{small}
\]

\[
mAP_{medium}
\]

\[
mAP_{large}
\]

based on clearly documented size definitions.

The validation set should ideally be split by mission/survey/location when metadata permit, rather than randomly mixing nearly adjacent frames.

---

# 31. Physics-Aware Validation

When true physical metadata are available, evaluate:

### Range consistency

\[
E_r=
|r_{pred}-r_{physical}|
\]

### Shadow consistency

\[
E_s=
|L_{shadow}^{pred}-L_{shadow}^{obs}|
\]

### Reconstruction error

\[
E_{recon}
=
\|I_{obs}-\hat I\|_1
\]

### Frequency consistency

\[
E_f=
\|
F_{LF}-F_{HF}
\|_1
\]

under matched target conditions.

---

# 32. Dataset-Specific Model Parameter Summary

Current EchoPhys-X-SSS640:

| Parameter | Value |
|---|---:|
| Input resolution | 640×640 |
| Input channels | 5 |
| Classes | 4 |
| Train images | 402 |
| Validation images | 110 |
| Train objects | 677 |
| Validation objects | 172 |
| Detection scales | 80×80 / 40×40 / 20×20 |
| Strides | 8 / 16 / 32 |
| FPN width | 128 |
| Backbone widths | 32 / 64 / 96 / 160 / 224 |
| Detection output | objectness + class + 4 LTRB |
| Parameter count | 1,110,875 |
| Attention | none / no global quadratic attention |
| Directional context | 1×7 + 7×1 depthwise mixing |
| Box activation | Softplus |
| Classification loss | Focal BCE |
| Box loss | Smooth L1 |
| Box-loss weight | 2.0 |
| Gaussian LF proxy radius | 2.2 |
| Local-contrast blur radius | 5.0 |
| Augmentation gain | 0.92–1.08 |
| Augmentation bias | -0.04–0.04 |
| Noise probability | 0.30 |
| Local attenuation probability | 0.20 |

These values correspond to the current dataset-specific implementation. fileciteturn3file0L22-L47

---

# 33. Code Snippet — Dataset Channels

```python
def make_sss_channels(im):
    lf = blur_np(im, 2.2)

    hf = np.clip(
        im - lf + 0.5,
        0.0, 1.0
    )

    local = np.abs(
        im - blur_np(im, 5.0)
    )

    local = np.clip(
        local * 3.0,
        0.0, 1.0
    )

    range_coord = np.repeat(
        np.linspace(
            0.0, 1.0,
            im.shape[1],
            dtype=np.float32
        )[None, :],
        im.shape[0],
        axis=0
    )

    return np.stack(
        [im, lf, hf, local, range_coord],
        axis=0
    ).astype(np.float32)
```

This is taken directly from the current dataset-specific implementation. fileciteturn3file0L63-L79

---

# 34. Code Snippet — Directional Mixer

```python
class LiteDirectionalMixer(nn.Module):
    def __init__(self, c):
        super().__init__()

        self.row = nn.Conv2d(
            c, c,
            kernel_size=(1, 7),
            padding=(0, 3),
            groups=c,
            bias=False
        )

        self.col = nn.Conv2d(
            c, c,
            kernel_size=(7, 1),
            padding=(3, 0),
            groups=c,
            bias=False
        )

        self.gate = nn.Sequential(
            nn.Conv2d(c, c, 1),
            nn.Sigmoid()
        )

        self.out = nn.Conv2d(
            c, c, 1, bias=False
        )

    def forward(self, x):
        r = self.row(x)
        c = self.col(x)
        g = self.gate(x)

        return x + self.out(
            g * r + (1 - g) * c
        )
```

The source explicitly labels this as a linear-cost directional mixer and not an official Mamba implementation. fileciteturn3file0L170-L182

---

# 35. Code Snippet — Detection Head

```python
class Head(nn.Module):
    def __init__(self, c=128, k=NUM_CLASSES):
        super().__init__()

        self.stem = DSConv(c, c)

        self.obj = nn.Conv2d(c, 1, 1)
        self.cls = nn.Conv2d(c, k, 1)
        self.box = nn.Conv2d(c, 4, 1)

    def forward(self, x):
        x = self.stem(x)

        return (
            self.obj(x),
            self.cls(x),
            F.softplus(self.box(x))
        )
```

This separates objectness from class prediction and avoids the incomplete `classes + 4` dense output used by the original HydroMamba scaffold. fileciteturn3file0L213-L222

---

# 36. Code Snippet — Loss

```python
def focal_bce(logits, target,
              gamma=2.0,
              alpha=0.25):

    p = torch.sigmoid(logits)

    ce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none"
    )

    pt = (
        p * target
        + (1 - p) * (1 - target)
    )

    at = (
        alpha * target
        + (1 - alpha) * (1 - target)
    )

    return (
        at
        * (1 - pt).pow(gamma)
        * ce
    ).mean()
```

Current box regression:

```python
lb = F.smooth_l1_loss(
    pred_box[mask],
    target_box[mask]
)

total = obj_loss + cls_loss + 2.0 * lb
```

The source implements this structure. fileciteturn3file0L264-L285

---

# 37. Code Snippet — Full Forward Pass

```python
class EchoPhysXDatasetOptimized(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Backbone()
        self.fpn = FPN()

        self.h3 = Head()
        self.h4 = Head()
        self.h5 = Head()

    def forward(self, x):

        p3, p4, p5 = self.backbone(x)

        f3, f4, f5 = self.fpn(
            p3, p4, p5
        )

        return {
            "p3": self.h3(f3),
            "p4": self.h4(f4),
            "p5": self.h5(f5)
        }
```

Source: current EchoPhys-X-SSS640 implementation. fileciteturn3file0L225-L234

---

# 38. Code Snippet — Testing

```python
from pathlib import Path
from echophys_x_dataset_optimized import sanity

root = Path("./dataset")

results = sanity(root)

print(results)
```

Command-line entry:

```bash
python echophys_x_dataset_optimized.py \
    --root ./dataset
```

The source script runs an input/shape/finite-loss/latency sanity check. fileciteturn3file0L288-L312

---

# 39. Full Research Algorithm

## Training

```text
1. Load SSS image + YOLO boxes.
2. Calibrate grayscale intensity.
3. Generate LF/HF proxy channels.
4. Generate local contrast and range coordinate.
5. Apply SSS-safe augmentation.
6. Pass 5-channel tensor to backbone.
7. Extract P3/P4/P5.
8. Fuse with FPN.
9. Predict objectness, class, LTRB.
10. Assign objects to scale-appropriate grids.
11. Calculate focal objectness loss.
12. Calculate focal classification loss.
13. Calculate box regression loss.
14. Backpropagate.
15. Validate on unseen images.
16. Repeat until convergence.
```

## Full future EchoPhys-X inference

```text
1. Acquire LF/HF sonar + metadata.
2. Estimate sound speed.
3. Estimate range and grazing geometry.
4. Correct propagation/range effects.
5. Estimate seabed response.
6. Estimate biological/biofouling field.
7. Fuse LF/HF representations.
8. Run multi-scale directional backbone.
9. Predict target, biology and shadow fields.
10. Estimate object geometry.
11. Predict theoretical shadow.
12. Compare predicted vs observed shadow.
13. Generate counterfactual de-biofouled representation.
14. Build target–biology–shadow graph.
15. Compute physical-consistency score.
16. Compute uncertainty.
17. Combine DL and physical evidence.
18. Reject/abstain when evidence is insufficient.
19. Return detection + class + geometry + confidence.
```

---

# 40. Complexity

For fixed-kernel directional convolutions:

\[
O(HWC)
\]

per layer.

The original attention operation:

\[
QK^T
\]

with \(Q,K\in\mathbb R^{H\times W\times d}\) creates:

\[
O(HW^2d)
\]

row-wise attention cost.

The supplied HydroMamba-V2 attention implementation explicitly creates the `[B,H,W,W]` tensor, so it is not globally linear. fileciteturn3file2L524-L534

EchoPhys-X-SSS640 eliminates this class of quadratic attention.

---

# 41. Computational Footprint

Current dataset-specific model:

\[
N_{params}=1,110,875
\]

Approximate FP32 parameter memory:

\[
1,110,875\times4
\approx4.24\text{ MB}
\]

This excludes:

- activations
- optimizer states
- runtime framework memory
- CUDA workspace
- input/output buffers

Therefore the 4.24 MB figure should not be described as total inference memory.

---

# 42. Real-Dataset Sandbox Results

The optimized architecture was run against actual validation images from the uploaded dataset.

Observed forward outputs:

\[
P3 = 80\times80
\]

\[
P4 = 40\times40
\]

\[
P5 = 20\times20
\]

The tested examples included:

- a scene containing 14 extremely small ground-truth objects,
- a medium/large example around 9% box area,
- a large example around 28% box area,
- a very large example around 69% box area.

Measured sandbox CPU forward times were approximately 92–175 ms/image for these individual tests.

These were **untrained-network forward tests**, not accuracy benchmarks.

No mAP claim was made.

---

# 43. What the Current Dataset Can and Cannot Prove

## It can support:

- four-class object detection
- multi-scale object detection
- small-object evaluation
- large-object evaluation
- grayscale SSS representation learning
- range-aware feature experiments
- hard-negative experiments
- objectness/class/box benchmarking

## It cannot currently validate:

- real dual-frequency fusion
- true Biot-Stoll inversion
- biological-cover classification
- biofouling fraction estimation
- shadow segmentation
- physics-derived target height
- material impedance estimation
- counterfactual de-biofouling accuracy

These require additional measurements/labels.

---

# 44. Required Next Dataset

For the complete EchoPhys-X research model, collect/annotate:

```text
target mask
shadow mask
biological-cover mask
seabed class
target class
visible fraction
biofouling fraction
burial fraction
orientation
estimated height
sonar altitude
range
frequency
temperature
salinity
depth
mission/survey ID
```

This enables the full multi-task objective.

---

# 45. Experimental Baselines

The final paper should compare EchoPhys-X with:

1. YOLO-family baseline
2. lightweight CNN detector
3. UNet/segmentation baseline where masks exist
4. Transformer/DETR-style baseline
5. state-space/Mamba-style baseline
6. EchoPhys-X-SSS640

Ablations should test:

```text
A. raw grayscale only
B. + LF/HF proxies
C. + local contrast
D. + range feature
E. + P3 small-object branch
F. + directional mixer
G. + hard-negative training
H. + uncertainty
I. full EchoPhys-X
```

---

# 46. Recommended Metrics

Detection:

\[
mAP_{50}
\]

\[
mAP_{50:95}
\]

\[
Precision
\]

\[
Recall
\]

\[
F1
\]

Small-object:

\[
mAP_s,\ Recall_s
\]

Medium:

\[
mAP_m,\ Recall_m
\]

Large:

\[
mAP_l,\ Recall_l
\]

Deployment:

\[
Latency_{ms}
\]

\[
FPS
\]

\[
Params
\]

\[
FLOPs
\]

Robustness:

- cross-mission performance
- cross-location performance
- cross-sonar performance
- low-SNR performance

Uncertainty:

- Expected Calibration Error (ECE)
- reliability diagram
- selective risk / coverage

---

# 47. Patent / Copyright Position

EchoPhys-X should not be described as guaranteed “patentable” or “completely novel.”

Current literature already contains:

- physics-guided SSS decomposition into interpretable fields,
- physics-informed target-shadow geometry,
- dual-frequency SSS fusion,
- small-target SSS detection using spatial/frequency features,
- deep-learning SSS detection and segmentation.

For example, PhysDNet proposes physics-guided decomposition of SSS into seabed reflectivity, terrain elevation and propagation loss. citeturn146010academia30

A 2026 physics-informed SSS detector uses target-height/shadow-length consistency to help distinguish weak targets from seabed rocks. citeturn146010search1

A 2026 dual-frequency SSS detection paper explicitly identifies the low-frequency/longer-range versus high-frequency/finer-detail tradeoff and combines dual-frequency and target-shadow reasoning. citeturn146010search4turn146010search6

Recent sonar small-target work also combines spatial/frequency features, multi-scale feature fusion and efficient attention. citeturn146010search5

Therefore the possible IP contribution should focus on the **specific implementation and interaction** of mechanisms, especially the biofouling-aware latent scene decomposition, counterfactual de-biofouling, target–biology–shadow graph reasoning and evidence-calibrated abstention, after a formal prior-art search.

Copyright can cover the software implementation and original source code. Patentability must be assessed by a qualified patent professional.

---

# 48. Important Scientific Claims to Avoid

Do not claim:

> “The current model performs real 300/900 kHz fusion.”

It does not on the uploaded dataset.

Do not claim:

> “The current model performs Biot-Stoll inversion.”

It does not.

Do not claim:

> “The current model automatically detects biofouled debris.”

There are no biofouling ground-truth masks in the uploaded dataset.

Do not claim:

> “The current model achieved X% accuracy.”

A trained benchmark has not yet been established.

Do not claim:

> “The model is guaranteed patentable.”

Patentability requires formal prior-art and claim analysis.

---

# 49. Recommended Final Research Configuration

For the uploaded dataset:

\[
\boxed{
EchoPhys\!-\!X\!-\!SSS640
}
\]

with:

```text
640×640 input
5 channels
4 classes
P3/P4/P5
80/40/20 feature maps
anchor-free detection
Focal BCE
Smooth-L1
depthwise-separable backbone
directional 1×7 / 7×1 mixing
range proxy
small-object emphasis
SSS-safe augmentation
```

For the full research version:

\[
\boxed{
EchoPhys\!-\!X
}
\]

with:

```text
measured LF/HF
+
environmental parameters
+
seabed estimation
+
biological-cover estimation
+
target estimation
+
shadow estimation
+
counterfactual de-biofouling
+
target–biology–shadow graph
+
physics consistency
+
uncertainty
+
abstention
```

---

# 50. Final Technical Definition

The complete EchoPhys-X research decision can be summarized as:

\[
\boxed{
\hat y
=
\arg\max_{y\in\mathcal Y}
P_\theta
\left(
y
\mid
I_{LF},
I_{HF},
G,
E,
B,
S_h
\right)
}
\]

subject to:

\[
E_{physics}<\tau_p
\]

\[
E_{shadow}<\tau_s
\]

\[
U<\tau_u
\]

otherwise:

\[
\boxed{
\hat y=\text{UNKNOWN / ABSTAIN}
}
\]

The central philosophy is:

> **EchoPhys-X should not classify a sonar patch only because it visually resembles debris. It should combine learned acoustic evidence with range/geometry, shadow consistency, biological-cover reasoning, multi-scale structure and uncertainty before declaring an anthropogenic object.**

---

# 51. Current Status

### Completed

- original HydroMamba architecture audit
- identification of quadratic attention issue
- identification of pseudo-physics issue
- dataset inspection
- dataset-specific five-channel input
- multi-scale P3/P4/P5 detector
- anchor-free detection formulation
- real-dataset forward simulation
- small/large-target analysis
- code for independent testing

### Not yet completed

- real training to convergence
- mAP benchmark
- real paired-frequency experiment
- biological/shadow annotation
- full physics reconstruction
- counterfactual de-biofouling validation
- cross-mission generalization study
- formal patent search/claim drafting

---

# 52. Recommended Immediate Experimental Roadmap

### Experiment 1
Train EchoPhys-X-SSS640 on all 402 training images.

### Experiment 2
Evaluate on 110 validation images.

### Experiment 3
Measure small/medium/large performance independently.

### Experiment 4
Add hard negatives and repeat.

### Experiment 5
Compare against a YOLO baseline.

### Experiment 6
Acquire/construct biofouled-debris annotations.

### Experiment 7
Add biology + shadow branches.

### Experiment 8
Acquire paired-frequency data.

### Experiment 9
Enable the complete physics layer.

### Experiment 10
Run a formal ablation and prior-art review before any public patent claim.

---

## References

- Uploaded HydroMamba-V2 technical audit and source code. fileciteturn3file2L360-L397
- Current EchoPhys-X-SSS640 implementation. fileciteturn3file0L22-L40
- Current five-channel dataset representation. fileciteturn3file0L63-L79
- Current P3/P4/P5 backbone and directional mixer. fileciteturn3file0L170-L195
- Current FPN and multi-scale heads. fileciteturn3file0L198-L234
- Current target assignment and detection loss. fileciteturn3file0L237-L285
- Current test/sanity entry point. fileciteturn3file0L288-L312
- PhysDNet: physics-guided SSS decomposition. citeturn146010academia30
- Physics-informed SSS target-shadow geometry. citeturn146010search1
- Dual-frequency SSS fusion and target-shadow pairing. citeturn146010search4turn146010search6
- Recent spatial/frequency small-target SSS detection. citeturn146010search5
- SSS propagation/sonar equation reference. citeturn146010search10
