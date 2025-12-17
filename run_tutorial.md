# BEVFusion Project Guide

This guide explains the key modules and file structure required to run the customized BEVFusion implementation for 3D object detection on LiDAR data.

## 1. Main Modules

The project is built on top of **MMDetection3D** and uses a specific **BEVFusion** implementation. Key components include:

### Core Framework
- **`external/mmdetection3d`**: The backbone library (OpenMMLab's MMDetection3D). It provides the core training, evaluation, and inference engines.
  - **`mmdet3d/`**: Contains the source code for detection algorithms, dataloaders, and model architectures.
  - **`projects/BEVFusion/`**: The specific BEVFusion implementation is housed here as a plugin/extension to mmdet3d.

### Custom Scripts
We have developed several utility scripts to simplify usage:
- **`setup_demo.py`**: The main interface for running inference. It handles dataset loading, config resolution, and executes the detector on sample files.
- **`visualize_demo.py`**: A dedicated visualization tool that takes the JSON predictions from `setup_demo.py` and renders high-quality BEV (Bird's Eye View) images with class-colored bounding boxes and Z-colored point clouds.
- **`graph_training_loss.py`**: Parses training logs to generate visualizations of Loss and mAP (mean Average Precision) over epochs.
- **`final_report.md`**: A comprehensive report documenting the training process, methodology, and results.

## 2. File Structure

To successfully run BEVFusion, the project relies on a specific directory layout.

### Root Directory (`3D-LiDAR-Based-Birds-Eye-View-Object-Detection-via-BEVFusion/`)

| Path | Description |
|------|-------------|
| `data/` | **CRITICAL**. Contains the dataset. Must follow the nuScenes format. |
| `external/` | Submodules, specifically `mmdetection3d`. |
| `work_dirs/` | Stores training outputs, logs, and **checkpoints** (e.g., `epoch_10.pth`). |
| `demo/` | Created by `setup_demo.py`. Contains sample inputs and **outputs** (visualizations). |
| `custom_changes/` | **Backup**. Contains copies of modified files from inside `external/` ensuring reproducibility. |

### Data Structure (`data/`)
The code expects standard nuScenes formatting. We use **Junctions** (on Windows) to map this structure to the actual dataset location to save space.

```text
data/
└── nuscenes/
    ├── maps/                   # Map expansion files
    ├── samples/                # Keyframes (LIDAR, CAMERA, RADAR)
    │   ├── LIDAR_TOP/          # We focus on this for LiDAR-only detection
    │   └── ...
    ├── sweeps/                 # Intermediate frames
    └── v1.0-mini/              # Metadata (JSONs)
```

### Configuration Files
The model configuration defines the architecture and training hyerparameters.
- **Location**: `external/mmdetection3d/projects/BEVFusion/configs/`
- **Main Config**: `bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`
  - Defines the VoxelNet backbone, SECONDFPN neck, and TransFusion head.
  - Key settings modified: `data_root`, `load_from` (checkpoint), `lr` (learning rate).

## 3. How to Run

### Run the Demo
To copy samples and run inference:
```bash
python setup_demo.py
```
*This generates predictions in `demo/outputs/preds`.*

### Visualize Results
To generate the images:
```bash
python visualize_demo.py
```
*Outputs images to `demo/outputs/vis`.*
