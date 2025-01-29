# Geometric Framework for 3D Cell Segmentation Correction

This repository contains the code implementation for the work:

*Geometric Framework for 3D Cell Segmentation Correction*

## Workflow

Welcome to GeoSeg! This the supplementary material that presents the implementation for the submission, which includes two pipeline: GeoSeg (for 2D-based method correction) and TiltedSeg(for 3D-based method correction). Please refer to each pipeline for further information on using GeoSeg.

## Main Pipeline - GeoSeg

## Repository Structure

```
.
├── data/                  # Folder containing GASP files from sections
├── cellstitch/           # Original CellStitch modules
├── CandidateSearching.py # Main program for candidate processing and stitching determination
├── EMD.py                # Sub-program for obtaining geometric information of candidates
├── Ochecker_reg.py       # Sub-program for obtaining topological shape of candidates
├── mlp_model_epoch_50.pth # Pre-trained model using labeled plant cells
├── tiltedseg.py          # Main program for filtering potential oversegmentation candidates
└── requirements.txt      # Required Python packages
```


## Sub Pipeline - TiltedSeg

A tool for correcting tilted oversegmentation produced by 3D-based segmentation methods.

## Repository Structure

```
.
├── data/                  # Folder containing 2D segmentation result, uploading your zarr result to here
├── cellstitch/           # Original CellStitch modules
├── CandidateSearching.py # Main program for candidate processing and stitching determination
├── EMD.py                # Sub-program for obtaining geometric information of candidates
├── Ochecker_reg.py       # Sub-program for obtaining topological shape of candidates
├── mlp_model_epoch_50.pth # Pre-trained model using labeled plant cells
├── tiltedseg.py          # Main program for filtering potential oversegmentation candidates
└── requirements.txt      # Required Python packages
```

## Pre-trained Model Information

The pre-trained model is built using labeled plant cells from CellPose 2D results. We strongly recommend:
- Building your own pre-trained model using your PlantSeg data
- Referring to our paper for detailed instructions on model building pipeline

## Human Feedback Integration

We strongly recommend integrating human feedback into the pipeline when using our pre-trained model on animal cells for the following reasons:

1. The pre-trained model is built upon CellStitch 2D segmentation results and transferred to the tilted case
2. The model is trained using plant cells and may not fully capture patterns present in animal or human cells

## Usage Pipeline

### Step 1: Data Preparation
Upload the "data" folder with your own cell 2D segmentation result folder from zarr. Maintain the folder name as "data".

### Step 2: Install Dependencies
```bash
cd titledseg
pip install -r requirements.txt
```

### Step 3: Run TiltedSeg
Execute with tolerance penalty and mode parameters:
```bash
python tiltedseg.py --PENALTY 1.3 --MODE minmax
```

Parameters:
- `PENALTY` (float, >0): Tolerance penalty
  - Larger values reduce the risk of incorrectly stitching non-oversegmented cells
- `MODE` (str, options: minmax or q1q3)
  - q1q3 mode provides tighter thresholds compared to minmax, reducing incorrect stitching risk

### Step 4: Review Results
Check the output for potential candidates in the format:
```
[[cell_ids1, cell_ids2], [..., ...]]
```
