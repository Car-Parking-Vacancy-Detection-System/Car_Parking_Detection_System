# Dataset Preparation

## Overview
This document provides an overview of the dataset preparation process for the car parking vacancy detection system. The steps involved include video format conversion, frame extraction, frame augmentation, and annotation.

## Steps

### 1. Convert Videos to Valid Format
Before processing, all videos were converted into a compatible format to ensure smooth handling.

### 2. Split Videos into Segments
- **Script:** `video_split.py`
- **Process:**
  - The videos were split into smaller segments of **30-minute** intervals using FFmpeg.
  - The resulting segments were stored in the `split_videos/` directory.
  - A summary of the splitting process was saved in `split_videos/splitting_summary.txt`.

### 3. Extract Frames
- **Script:** `frame_extraction.py`
- **Process:**
  - Frames were extracted from the video segments at an interval of **15 frames**.
  - Frames were filtered based on distinctiveness and blurriness.
  - Extracted frames were stored in `extracted_frames/`.
  - A summary was saved in `extracted_frames/extraction_summary.txt`.

### 4. Extract Distinct Frames
- Extracted frames were further processed to remove near-duplicate frames using a threshold method.

### 5. Augment Frames
- **Script:** `augmentation.py`
- **Process:**
  - Augmentations applied: Horizontal flip, rotation, brightness/contrast adjustments, Gaussian blur.
  - Each frame generated **5 augmented versions**.
  - Augmented frames were saved in `new_augmented_dataset/`.
  - A summary was saved in `new_augmentation_summary.txt`.

### 6. Annotate Frames
- **Script:** `auto_annotator.ipynb`
- **Process:**
  - Initial annotation was done via an automated script.
  - Manual corrections were applied to improve accuracy.
  - A total of **2500 frames** were annotated.
