# OptimalColloidNet

A deep learning pipeline for detecting and tracking colloidal particles in microscopy video. The model is trained entirely on **synthetic data** — no labeled real microscopy images are required. It outputs per-frame particle center coordinates with sub-pixel accuracy via a learned offset field.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [How It Works](#how-it-works)
  - [Synthetic Data Generation](#synthetic-data-generation)
  - [Model Architecture](#model-architecture)
  - [Loss Functions](#loss-functions)
  - [Inference Pipeline](#inference-pipeline)
- [Kaggle Setup](#kaggle-setup)
  - [Step 1 — Create a New Notebook](#step-1--create-a-new-notebook)
  - [Step 2 — Install Dependencies](#step-2--install-dependencies)
  - [Step 3 — Run Training](#step-3--run-training)
  - [Step 4 — Run Video Inference](#step-4--run-video-inference)
- [File Reference](#file-reference)
- [Hyperparameters](#hyperparameters)
- [Outputs](#outputs)
- [Tips for Kaggle](#tips-for-kaggle)
- [Requirements](#requirements)

---

## Overview

| Property | Value |
|---|---|
| Task | Particle detection in microscopy video |
| Input | Grayscale microscopy video (`.mp4`) |
| Output | Annotated video with per-frame particle count and dot overlays |
| Training data | Fully synthetic — generated on-the-fly |
| Modalities supported | Phase contrast, darkfield, fluorescence, DIC, brightfield |
| Sub-pixel accuracy | Yes — via learned offset displacement field |
| Test-time augmentation | 8-fold (2 rotations × 2 horizontal flips × 2 vertical flips) |
| GPU support | Single and multi-GPU; mixed precision (AMP) on CUDA |

---

## Repository Structure

```
.
├── optimalcolloidpython.ipynb   # Full training pipeline (run this on Kaggle first)
├── analyze_video.py             # Inference-only script for processing real video
├── Requirements.txt             # Python dependencies
└── README.md                    # This file
```

After training, a `colloid_output/` directory is created containing:

```
colloid_output/
├── best_checkpoint.pt           # Saved model weights (used by analyze_video.py)
├── training_curve.png           # Loss and F1 curves
└── result_animation.gif         # Inference result on a synthetic test video
```

---

## How It Works

### Synthetic Data Generation

The model is trained on procedurally generated images — no real labeled microscopy data is needed. Each synthetic image is rendered by:

1. **Placing particles** randomly on a canvas (`IMAGE_SIZE = 256`), with radii uniformly sampled between 6 and 14 pixels. Up to 30% of samples force touching/nearly-overlapping particles (controlled by `HARD_EXAMPLE_FRAC = 0.30`) to improve robustness.

2. **Rendering each particle** into its own layer using a physically-motivated appearance model for each microscopy modality. The five supported modalities and their sampling probabilities are:

   | Modality | Probability | Appearance model |
   |---|---|---|
   | Phase contrast | 50% | Dark ring + bright center + fringes |
   | Darkfield | 20% | Bright ring on dark background |
   | Fluorescence | 15% | Additive Gaussian PSF blob |
   | DIC | 10% | Shadow-relief gradient effect |
   | Brightfield | 5% | Low-contrast transmission dip |

3. **Compositing layers** onto a randomized background with vignetting and slow-varying illumination gradients. Additive modalities (fluorescence, darkfield) sum contributions; multiplicative modalities use combined delta compositing to correctly model overlapping particle saddle points.

4. **Adding realistic noise**: Poisson shot noise (photon scale 200–600), Gaussian readout noise (σ = 0.015), and random hot pixels.

5. **Generating ground-truth targets** for three prediction heads:
   - **Heatmap**: Additive Gaussian blobs (σ = radius/2.5) centered at each particle — additive rather than max-pooled so overlapping particles each preserve their own peak.
   - **Mask**: Binary circle mask for each particle.
   - **Offset map**: Per-pixel vector pointing to the nearest particle center in exact pixel units, used for sub-pixel refinement at inference.

The dataset uses a 70/15/15 train/val/test split over 2000 generated images.

---

### Model Architecture

`OptimalColloidNet` is a U-Net–style encoder-decoder with attention gates and an ASPP bottleneck.

```
Input (1×H×W grayscale)
    │
    ▼
Stem: ConvBnRelu(1→32) + ResBlock(32)
    │
    ├─ Encoder
    │   enc1: MaxPool → ConvBnRelu(32→64)  + ResBlock(64)
    │   enc2: MaxPool → ConvBnRelu(64→128) + ResBlock(128) × 2
    │   enc3: MaxPool → ConvBnRelu(128→256)+ ResBlock(256) × 2
    │   enc4: MaxPool → ConvBnRelu(256→512)+ ResBlock(512, drop=0.15) × 2
    │
    ├─ Bottleneck
    │   ASPP(512→512, dilations=[1,2,4,8]) + global average pooling branch
    │
    └─ Decoder (with Attention Gates on each skip connection)
        dec4: UpBlock(512→256) + AttentionGate(enc3) → ConvBnRelu + ResBlock
        dec3: UpBlock(256→128) + AttentionGate(enc2) → ConvBnRelu + ResBlock
        dec2: UpBlock(128→64)  + AttentionGate(enc1) → ConvBnRelu + ResBlock
        dec1: UpBlock(64→32)   + AttentionGate(stem) → ConvBnRelu + ResBlock
            │
            ├─ refine_hmap   → heatmap_head   → (1×H×W) logits
            ├─ refine_mask   → mask_head      → (1×H×W) logits
            └─ refine_offset → offset_head    → (2×H×W) dx/dy displacements
```

**Key design choices:**
- **ResBlocks** use pre-activation (BN → ReLU → Conv) with 10% dropout inside residual branches (15% in enc4).
- **Attention Gates** suppress irrelevant activations on skip connections; gating signal comes from the decoder stream.
- **ASPP bottleneck** captures multi-scale context with parallel dilated convolutions (rates 1, 2, 4, 8) plus a global average pooling branch.
- **Three output heads** share the same decoder feature map but each has its own refinement block (ConvBnRelu → ResBlock → ConvBnRelu).
- **Bias initialization**: heatmap and mask head final convolutions are initialized to −2.19 (≈ sigmoid⁻¹(0.1)) to counteract class imbalance in sparse particle fields.

---

### Loss Functions

Total loss is a weighted sum of three terms:

```
L_total = 1.0 × L_heatmap + 1.0 × L_mask + 0.5 × L_offset
```

| Head | Loss | Details |
|---|---|---|
| Heatmap | Modified Focal Loss | α = 2.0, β = 4.0. Positive pixels are strict center locations only (target == 1.0). Negative penalty weighted by `(1 − target)^β` to down-weight near-center pixels. Normalized by number of positive pixels. |
| Mask | Dice + BCE | Standard combination; Dice computed per-image in the batch, averaged. Smooth = 1.0. |
| Offset | Masked Smooth L1 | Loss computed only within particles (ground-truth offset magnitude > 0.01 px). Normalized by number of active pixels. |

Training uses:
- **Optimizer**: AdamW, lr = 3×10⁻⁴, weight decay = 1×10⁻⁴
- **Scheduler**: CosineAnnealingWarmRestarts, T₀ = 20 epochs, η_min = 3×10⁻⁷; stepped at fractional epoch granularity (per batch)
- **Gradient clipping**: max norm = 1.0
- **Mixed precision**: `torch.amp.GradScaler` on CUDA; scheduler step only fires when the loss scale did not decrease (skipped steps are dropped)
- **Early stopping**: patience = 8 epochs on validation loss

---

### Inference Pipeline

`analyze_video.py` processes a video file frame by frame:

1. **Auto-scale detection** (first frame only): Hough circle transform estimates the median particle radius in pixels. A scale factor is computed as `target_radius (10px) / median_radius`, clamped to [0.25, 4.0]. This rescales particles to the radius range the network was trained on.

2. **Preprocessing**: Frame is converted to grayscale, optionally resized by the scale factor, normalized to [0, 1], and zero-padded to a multiple of 16 (required by the 4× pooling depth of the encoder).

3. **8-fold TTA inference**: The padded image is augmented into 8 variants (rot90×{0,1} × hflip×{F,T} × vflip×{F,T}), run through the network in a single batched forward pass, then un-augmented and averaged. The offset field reversal applies the correct rotation formula (`(dy, dx) → (−dx, dy)` per 90° CCW rotation) and sign flips for each spatial flip.

4. **Peak detection**: `skimage.feature.peak_local_max` finds heatmap peaks above threshold 0.15 with minimum distance 3 px.

5. **Sub-pixel refinement**: Each integer peak `(r, c)` is refined by reading `offset_map[:, r, c]` = `(dy, dx)` and computing the final center as `(r + dy, c + dx)`.

6. **Coordinate back-projection**: Refined centers (in scaled+padded space) are validated against the pre-padding dimensions to discard padding artifacts, then divided by `scale_factor` to recover original video coordinates.

7. **Annotation**: Green filled dot (r=3) with dark green outline (r=5) drawn at each detected center. Particle count and scale factor overlaid in top-left corner.

---

## Kaggle Setup

### Step 1 — Create a New Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Under **Settings** (right panel) → set **Accelerator** to **GPU T4 × 2** or **P100** for best performance. The notebook supports multi-GPU automatically via `DataParallel`.
3. Set **Persistence** → **Variables and Files** so outputs survive between sessions.
4. Upload all three files (`optimalcolloidpython.ipynb`, `analyze_video.py`, `Requirements.txt`) via **File → Upload** or attach as a dataset.

### Step 2 — Install Dependencies

Add a code cell at the top of your notebook:

```python
import subprocess
subprocess.run(["pip", "install", "-r", "/kaggle/input/<your-dataset>/Requirements.txt", "-q"])
```

Or install directly:

```python
!pip install torch torchvision opencv-python numpy scipy scikit-image matplotlib pillow -q
```

> **Note**: PyTorch and torchvision are pre-installed on Kaggle GPU kernels. The above installs the remaining packages. If you see version conflicts, the pre-installed versions are sufficient.

### Step 3 — Run Training

Open `optimalcolloidpython.ipynb` and run all cells from top to bottom. The notebook is self-contained — it generates all training data on-the-fly, trains the model, and saves outputs.

**Expected runtime on Kaggle:**

| Hardware | Approx. time |
|---|---|
| GPU T4 × 1 | ~35–50 min (60 epochs) |
| GPU T4 × 2 | ~20–30 min (batch size doubled automatically) |
| GPU P100 | ~25–40 min |
| CPU only | Not recommended (several hours) |

Training will produce the following in `colloid_output/` (accessible at `/kaggle/working/colloid_output/`):

- `best_checkpoint.pt` — weights at the epoch with lowest validation loss
- `training_curve.png` — train/val loss and val F1 plotted over epochs
- `result_animation.gif` — 25-frame synthetic test video with heatmap and detected centers

Early stopping triggers if validation loss does not improve for 8 consecutive epochs, so actual runtime may be shorter than the 60-epoch maximum.

### Step 4 — Run Video Inference

Once training is complete and `best_checkpoint.pt` exists:

```python
import subprocess
subprocess.run([
    "python", "/kaggle/input/<your-dataset>/analyze_video.py"
])
```

Or copy the script and modify the paths directly in a notebook cell:

```python
# At the bottom of analyze_video.py, update:
INPUT_VIDEO   = "/kaggle/input/<your-video-dataset>/sample_microscopy.mp4"
MODEL_WEIGHTS = "/kaggle/working/colloid_output/best_checkpoint.pt"
OUTPUT_VIDEO  = "/kaggle/working/annotated_output.mp4"
```

The annotated output video will be saved to `/kaggle/working/annotated_output.mp4` and visible in the output panel.

---

## File Reference

### `optimalcolloidpython.ipynb`

The full end-to-end training notebook. Sections in order:

| Section | Description |
|---|---|
| 1. Parameters | All hyperparameters and constants in one place |
| 2. Multi-type Microscopy Renderer | Per-particle layer rendering for all 5 modalities |
| 3. Ground-truth Helpers | Gaussian heatmap, binary mask, and offset map generation |
| 4. Background Generation | Randomized illumination gradients, vignetting, and noise |
| 5. Synthetic Sample Generation | Assembles full images with Poisson + Gaussian noise |
| 6. Dataset Augmentation | Online augmentation (flips, rotations, contrast jitter) |
| 7. Model Architecture | `OptimalColloidNet` definition |
| 8. Loss Functions | Focal loss, Dice+BCE, masked Smooth L1 |
| 9. Detection | Peak finding + offset-field sub-pixel refinement |
| 10. Detection Metrics | Precision, recall, F1 via Hungarian matching |
| 11. Training | Full training loop with AMP, grad clipping, early stopping |
| 12. TTA Inference | 8-fold test-time augmentation |
| 13. Test Video | Synthetic Brownian-motion video generation + inference |
| 14. Training Curves | Saved matplotlib figure |
| 15. Animation | GIF of inference results |

### `analyze_video.py`

Standalone inference script. Takes a real microscopy video, applies the trained model frame by frame, and writes an annotated output video. Does not require the training notebook to be in the same environment — only `best_checkpoint.pt` is needed.

### `Requirements.txt`

```
torch
torchvision
opencv-python
numpy
scipy
scikit-image
matplotlib
pillow
```

---

## Hyperparameters

All training hyperparameters are defined at the top of `optimalcolloidpython.ipynb` under **Section 1 — Parameters**.

| Parameter | Value | Description |
|---|---|---|
| `IMAGE_SIZE` | 256 | Training image resolution (px) |
| `MIN_RADIUS` | 6 | Minimum synthetic particle radius (px) |
| `MAX_RADIUS` | 14 | Maximum synthetic particle radius (px) |
| `NUM_PARTICLES` | 60 | Particles per synthetic image |
| `NUM_IMAGES` | 2000 | Total synthetic images generated |
| `BATCH_SIZE` | 8 | Per-GPU batch size (doubled automatically with 2 GPUs) |
| `MAX_EPOCHS` | 60 | Maximum training epochs |
| `PATIENCE` | 8 | Early stopping patience (val loss) |
| `HMAP_W` | 1.0 | Heatmap loss weight |
| `MASK_W` | 1.0 | Mask loss weight |
| `OFFSET_W` | 0.5 | Offset loss weight |
| `DETECT_THRESHOLD` | 0.15 | Heatmap peak detection threshold |
| `NMS_MIN_DIST` | 3 | Minimum pixel distance between detections |
| `HARD_EXAMPLE_FRAC` | 0.30 | Fraction of images with forced touching particles |
| Learning rate | 3×10⁻⁴ | AdamW initial LR |
| Weight decay | 1×10⁻⁴ | AdamW weight decay |
| LR schedule | CosineAnnealingWarmRestarts | T₀=20, η_min=3×10⁻⁷ |
| Grad clip norm | 1.0 | Max gradient norm |
| Train / Val / Test split | 70% / 15% / 15% | Of `NUM_IMAGES` |

**Inference-specific parameters** (in `analyze_video.py`):

| Parameter | Value | Description |
|---|---|---|
| `target_radius` | 10.0 px | Target radius the model expects; drives auto-scaling |
| `scale_factor` clamp | [0.25, 4.0] | Prevents extreme rescaling on unusual inputs |
| `threshold` | 0.15 | Same as `DETECT_THRESHOLD` |
| `min_dist` | 3 | Same as `NMS_MIN_DIST` |

---

## Outputs

### From the training notebook (`colloid_output/`)

| File | Description |
|---|---|
| `best_checkpoint.pt` | PyTorch checkpoint dict: `{epoch, model_state, optimizer_state, val_loss}` |
| `training_curve.png` | Train loss, val loss, and val F1 over epochs |
| `result_animation.gif` | Side-by-side: raw synthetic frame vs. predicted mask + detected centers |

### From `analyze_video.py`

| File | Description |
|---|---|
| `annotated_output.mp4` | Input video with green detection dots and particle count overlay |

---

## Tips for Kaggle

**Saving outputs between sessions**: Kaggle working directory (`/kaggle/working/`) persists within a session but is wiped when the kernel is reset. To preserve `best_checkpoint.pt` across sessions, either download it from the output panel or save it to a Kaggle dataset via the output panel → **Save Version**.

**Resuming training**: The checkpoint saves optimizer state alongside model weights, but the notebook does not implement resume-from-checkpoint out of the box. To resume, load `best_checkpoint.pt` and restore both `model_state` and `optimizer_state` before the training loop.

**Scaling to larger particles**: If your real video has particles significantly larger or smaller than 6–14 px (at the model's native 256 px scale), the auto-scale logic in `analyze_video.py` will handle rescaling automatically at inference. For best results during training, consider adjusting `MIN_RADIUS` and `MAX_RADIUS` to match your expected particle size range before rescaling.

**Memory**: The model has approximately 8M parameters. At `BATCH_SIZE = 8` and `IMAGE_SIZE = 256`, peak GPU memory during training is roughly 4–6 GB. A Kaggle T4 (16 GB) has comfortable headroom. If you increase `IMAGE_SIZE`, reduce `BATCH_SIZE` proportionally.

**Improving detection on real data**: If the model misses particles or generates false positives on your specific microscopy modality, the most effective lever is to adjust `RENDER_TYPE_PROBS` in the notebook to over-represent the modality closest to your data, then retrain.

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12 (≥ 2.0 recommended; required for `weights_only=True` in `torch.load`)
- CUDA-capable GPU strongly recommended for training; inference runs on CPU but is slow

See `Requirements.txt` for the full package list.
