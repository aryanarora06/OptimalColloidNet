# OptimalColloidNet: Microscopic Particle Detection

## Overview
OptimalColloidNet is a deep learning framework designed to accurately detect and localize colloidal particles in microscopy videos. Utilizing a custom multi-head Convolutional Neural Network (CNN) featuring Attention Gates and ASPP, the model predicts exact sub-pixel particle centers.

This repository contains the complete pipeline: from synthetic multi-modal microscopy data generation and model training to high-precision test-time augmented (TTA) inference on real videos.

**Note:** This guide is specifically tailored for running the pipeline on **Kaggle**.

## Kaggle Environment Setup

Kaggle provides free GPU access, which is highly recommended for training this model efficiently.

1. **Create a New Notebook:** Go to Kaggle, click **Create**, and select **Notebook**.
2. **Upload Code:** Go to **File > Import Notebook** and upload the `optimalcolloidpython.ipynb` file.
3. **Enable GPU:** In the panel under **Settings**, set the **Accelerator** to **GPU T4 x2** or **GPU P100**.
4. **Enable Internet:** Ensure **Internet** is toggled **On** (required if you need to `pip install` any specific versions of packages, though Kaggle has most pre-installed).

## Usage Guide

### 1. Training the Model
Simply run all cells in the `optimalcolloidpython.ipynb` notebook. 
* The notebook will automatically generate 2,000 synthetic multi-modal images and train the model for 60 epochs.
* Outputs (including `best_checkpoint.pt`, training curve plots, and an evaluation GIF) will be saved to Kaggle's working directory at `/kaggle/working/colloid_output/`.

### 2. Running Video Inference
To run the `analyze_video.py` script on Kaggle, you will need to upload your target video as a Kaggle Dataset.

1. **Upload your video:** Click **Add Input** (or **Add Data**) in the right panel -> **New Dataset** -> Upload your `sample_microscopy.mp4` file.
2. **Modify Paths:** Since Kaggle uses a specific file structure, you will need to update the execution paths at the bottom of `analyze_video.py` (or modify them dynamically in a notebook cell).

```python
if __name__ == '__main__':
    # Point to your uploaded dataset directory
    INPUT_VIDEO  = "/kaggle/input/your-dataset-name/sample_microscopy.mp4" 
    
    # Point to the trained weights in the working directory
    MODEL_WEIGHTS = "/kaggle/working/colloid_output/best_checkpoint.pt"
    
    # Save the output to the working directory so you can download it
    OUTPUT_VIDEO = "/kaggle/working/annotated_output.mp4"
    
    process_video(INPUT_VIDEO, MODEL_WEIGHTS, OUTPUT_VIDEO)
```

3. **Execute the script:** You can upload `analyze_video.py` to your Kaggle workspace and run it directly from a notebook cell using the `!` command:
```bash
!pip install -r Requirements.txt  # Optional: to ensure exact dependencies match
!python analyze_video.py
```

### 3. Downloading Your Results
Once the script finishes processing the video, look at the **Output** section in the right-hand panel of your Kaggle notebook. You will see `annotated_output.mp4` and the `colloid_output/` folder. Click the three dots next to any file to download it directly to your local machine.
