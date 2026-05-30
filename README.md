# Colloid Particle Detection

This project consists of two parts:

- **`optimalcolloidpython.ipynb`** — trains the `OptimalColloidNet` model on synthetically generated microscopy images and saves a checkpoint.
- **`analyze_video.py`** — loads the trained checkpoint and runs inference on a real microscopy video, producing an annotated output video.

---

## Step 1 — Set Up a Kaggle Notebook

1. Go to [kaggle.com](https://www.kaggle.com) and sign in.
2. Click **+ New Notebook**.
3. In the top-right, click the ⚙️ settings panel and set **Accelerator** to **GPU T4 x2** (recommended) or GPU P100.
4. Make sure **Internet** is toggled **on** (required for any pip installs).

---

## Step 2 — Upload Your Files

In the notebook sidebar, click **Add data → Upload** and add:

- `optimalcolloidpython.ipynb`
- `analyze_video.py`
- Your microscopy video file (e.g. `sample_microscopy.mp4`)

Uploaded files land at `/kaggle/input/<dataset-name>/`.

---

## Step 3 — Install Dependencies

All major libraries are pre-installed on Kaggle. Add this cell at the top of your notebook to confirm/install any missing ones:

```python
!pip install -q scikit-image opencv-python-headless
```

The full dependency list used by both files:

| Library | Notes |
|---|---|
| `torch` / `torchvision` | Pre-installed on Kaggle GPU images |
| `numpy` | Pre-installed |
| `opencv-python-headless` | Use the headless build on Kaggle |
| `scipy` | Pre-installed |
| `scikit-image` | May need install (provides `peak_local_max`) |
| `matplotlib` | Pre-installed |

---

## Step 4 — Train the Model (run the notebook)

Open `optimalcolloidpython.ipynb` and run all cells. Training will:

- Synthetically generate 2000 microscopy images on the fly.
- Train for up to 60 epochs with early stopping (patience = 8).
- Save the best checkpoint to `colloid_output/best_checkpoint.pt`.

Expected training time on a single T4 GPU: ~20–40 minutes.

---

## Step 5 — Run Inference on Your Video

Once training is done, run `analyze_video.py`. Update the three path variables at the bottom of the script before executing:

```python
INPUT_VIDEO   = "/kaggle/input/<your-dataset>/sample_microscopy.mp4"
MODEL_WEIGHTS = "colloid_output/best_checkpoint.pt"
OUTPUT_VIDEO  = "/kaggle/working/annotated_output.mp4"
```

Then run from a notebook cell:

```python
%run analyze_video.py
```

Or from a terminal cell:

```bash
!python analyze_video.py
```

The script will:

1. Auto-detect particle scale from the first frame.
2. Run 8-fold test-time augmentation (TTA) per frame.
3. Write an annotated video with detected particle centres overlaid in green.

Output is saved to `/kaggle/working/annotated_output.mp4`, which you can download directly from the Kaggle file browser on the right-hand panel.

---

## Output Files

| File | Location | Description |
|---|---|---|
| `best_checkpoint.pt` | `colloid_output/` | Trained model weights |
| `loss_curve.png` | `colloid_output/` | Training/validation loss plot |
| `annotated_output.mp4` | `/kaggle/working/` | Video with particle detections |

---

## Tips

- **GPU is required.** The model will fall back to CPU but inference will be very slow.
- **Out of memory?** Reduce `BATCH_SIZE` in the notebook (default is 8; try 4).
- **No particles detected?** Lower `DETECT_THRESHOLD` in `analyze_video.py` (default `0.15`; try `0.10`).
- **Wrong particle size?** The `estimate_optimal_scale` function auto-scales based on Hough circle detection on the first frame. If it fails, set `scale_factor` manually in `process_video()`.
