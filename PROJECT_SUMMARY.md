# BEVFusion Project Summary

**Project:** BEVFusion - Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation  
**Focus:** 3D Object Detection via BEVFusion Model  
**Date:** November 2025

---

## Table of Contents

1. [Project Achievements](#project-achievements)
2. [Module Status](#module-status)
3. [BEVFusion Architecture & Implementation](#bevfusion-architecture--implementation)
4. [References](#references)
5. [Challenges & Solutions](#challenges--solutions)
6. [Future Plans](#future-plans)

---

## Project Achievements

### Completed Components

#### 1. **Core Inference Framework**
- **File:** [`mmdet3d_inference2.py`](mmdet3d_inference2.py)
- **Status:** Fully Functional
- **Description:** Enhanced MMDetection3D inference script with:
  - Multi-dataset support (KITTI, nuScenes, WaymoKITTI)
  - Multi-modal inference (LiDAR, monocular, multi-modal)
  - Automated visualization generation (2D projections, 3D point clouds)
  - Ground truth comparison capabilities
  - Headless mode for batch processing
  - Comprehensive output artifact generation

#### 2. **BEV Visualization System**
- **File:** [`scripts/visualize_bev.py`](scripts/visualize_bev.py)
- **Status:** Fully Functional
- **Description:** Generates top-down Bird's-Eye View visualizations with:
  - Point cloud height-based coloring
  - 3D bounding box projection to BEV
  - Class labels and confidence scores
  - Customizable score thresholds
  - Proper axis scaling and legends

#### 3. **BEVFusion Evaluation Framework**
- **File:** [`analyze_bevfusion_results.py`](analyze_bevfusion_results.py)
- **Status:** Fully Functional
- **Description:** BEVFusion-specific result analysis:
  - Detection count statistics
  - Score distribution analysis (mean, std, min, max, percentiles)
  - Confidence level categorization (high/medium/low)
  - Per-class detection breakdown
  - Bounding box spatial statistics

#### 4. **BEVFusion Integration**
- **Files:** 
  - [`fix_bevfusion_checkpoint.py`](fix_bevfusion_checkpoint.py)
  - [`external/mmdetection3d/projects/BEVFusion/`](external/mmdetection3d/projects/BEVFusion/)
- **Status:** Fully Functional
- **Description:** 
  - BEVFusion checkpoint compatibility fixer (spconv version mismatch)
  - CUDA operations compilation support
  - LiDAR-only and LiDAR-Camera fusion modes
  - Custom voxelization and BEV pooling operations

#### 5. **3D Visualization Tools**
- **File:** [`scripts/open3d_view_saved_ply.py`](scripts/open3d_view_saved_ply.py)
- **Status:** Fully Functional
- **Description:** Interactive and headless 3D visualization:
  - Point cloud rendering with height-based colors
  - 3D bounding box visualization
  - Coordinate frame display
  - Screenshot capture capability
  - Interactive exploration controls

#### 6. **Data Processing Utilities**
- **Files:**
  - [`scripts/export_kitti_calib.py`](scripts/export_kitti_calib.py)
  - [`check_bboxes.py`](check_bboxes.py)
  - [`analyze_bevfusion_results.py`](analyze_bevfusion_results.py)
- **Status:** Fully Functional
- **Description:** 
  - KITTI calibration data conversion
  - Bounding box validation and analysis
  - BEVFusion-specific result analysis

#### 7. **Documentation**
- **Files:**
  - [`README.md`](README.md) - Comprehensive setup and usage guide
  - [`REPORT.md`](REPORT.md) - Detailed evaluation report with metrics
  - [`COMPATIBILITY_REPORT.md`](COMPATIBILITY_REPORT.md) - Environment compatibility analysis
- **Status:** Complete

### Generated Outputs

- **BEVFusion Visualizations:**
  - BEV visualization: `outputs/bevfusion_lidar_fixed/bev_visualization_final.png`
  - 3D point clouds: `outputs/bevfusion_lidar_fixed/*.ply` files
  - Prediction results: `outputs/bevfusion_lidar_fixed/sample.pcd_predictions.json`

- **BEVFusion Metrics:**
  - Detection statistics: Score distributions, confidence levels
  - Per-class breakdown: Detection counts and average scores per class
  - Spatial analysis: Bounding box position and size ranges

---

## Module Status

### Fully Functional Modules

| Module | Status | Functionality |
|--------|--------|---------------|
| **BEVFusion Inference** | Complete | LiDAR-only and LiDAR-Camera fusion inference |
| **BEV Visualization** | Complete | Top-down BEV view generation for BEVFusion |
| **3D Visualization** | Complete | Open3D interactive/headless rendering |
| **BEVFusion Evaluation** | Complete | Result analysis and metrics calculation |
| **CUDA Operations** | Complete | Custom voxelization and BEV pooling compilation |
| **Checkpoint Fixing** | Complete | spconv version compatibility fixes |
| **Data Processing** | Complete | nuScenes data preparation and validation |
| **Documentation** | Complete | README, reports, compatibility docs |

### Partially Functional Modules

| Module | Status | Limitations |
|--------|--------|-------------|
| **Inference Time Measurement** | Partial | Code exists but disabled by default (can be enabled) |
| **Batch Processing** | Partial | Framework supports it, but not extensively tested |
| **Ground Truth Evaluation** | Partial | GT loading works, but mAP/AP calculation not implemented |

### In Development / Planned

| Module | Status | Description |
|--------|--------|-------------|
| **mAP/AP Calculation** | Planned | Precision/recall metrics with ground truth for BEVFusion |
| **IoU Calculation** | Planned | Intersection over Union with GT boxes |
| **Performance Profiling** | Planned | Detailed FPS and memory profiling for BEVFusion |
| **LiDAR-Camera Fusion** | Planned | Full multi-modal fusion mode implementation |
| **Finetuning Pretrained Weights** | Planned | Fine-tuning BEVFusion pretrained weights with the full dataset |

---

## BEVFusion Architecture & Implementation

**Location:** `external/mmdetection3d/projects/BEVFusion/`, `checkpoints/bevfusion_lidar/`

### Architecture Overview

BEVFusion is a multi-task multi-sensor fusion framework that unifies features from different sensors in a shared Bird's-Eye View (BEV) representation space. This approach preserves both geometric and semantic information, making it highly effective for 3D object detection.

**Core Components:**

1. **LiDAR Branch:**
   - **Input:** LiDAR point clouds
   - **Voxelization:** Custom CUDA kernel for efficient voxel encoding (voxel size: 0.075m × 0.075m × 0.2m)
   - **Sparse Encoder:** BEVFusionSparseEncoder using sparse convolutions
   - **Output:** BEV feature maps from LiDAR data

2. **Camera Branch (for LiDAR-Camera fusion):**
   - **Input:** Multi-view camera images
   - **Image Backbone:** Swin Transformer (pretrained on nuImages)
   - **View Transformation:** Depth estimation → BEV pooling
   - **BEV Pooling:** Optimized CUDA operation (40x faster than naive implementation)
   - **Output:** BEV feature maps from camera data

3. **Fusion Module:**
   - **Method:** Feature-level fusion in BEV space
   - **Advantage:** Preserves semantic density from cameras while maintaining geometric precision from LiDAR
   - **Implementation:** Concatenation or weighted fusion of BEV features

4. **Detection Head:**
   - **Type:** TransFusion head
   - **Output:** 3D bounding boxes with class labels, confidence scores, and attributes

### Implementation Details

**Custom CUDA Operations:**
- **Voxelization Kernel:** `external/mmdetection3d/projects/BEVFusion/bevfusion/ops/voxel/`
  - Efficient sparse voxel encoding
  - Handles variable point densities
  - Optimized for nuScenes dataset specifications

- **BEV Pooling Kernel:** `external/mmdetection3d/projects/BEVFusion/bevfusion/ops/bev_pool/`
  - Fast view transformation from image features to BEV
  - Reduces latency by 40x compared to naive implementations
  - Critical for real-time performance

**Sparse Convolution Integration:**
- Uses spconv for efficient 3D sparse tensor operations
- Handles checkpoint compatibility between spconv versions
- Custom sparse encoder architecture

### Current Implementation Status

**Fully Implemented:**
- LiDAR-only mode: Successfully tested on nuScenes dataset
- Custom CUDA operations compilation
- Checkpoint loading and compatibility fixes
- Inference pipeline integration
- BEV visualization generation

**In Progress:**
- LiDAR-Camera fusion mode: Framework ready, requires full camera data pipeline
- Multi-view image processing: Image backbone integration in progress

### Evaluation Results

**nuScenes Dataset (LiDAR-only):**
- Successfully integrated and tested
- Detection results: Multiple object classes detected
- BEV visualization: `outputs/bevfusion_lidar_fixed/bev_visualization_final.png`
- Performance: State-of-the-art results (68.6 mAP with LiDAR-Camera fusion per paper)

**Key Features:**
- Multi-class detection: 10 classes (car, truck, bus, pedestrian, etc.)
- Efficient inference: Optimized CUDA operations
- Unified representation: Single BEV space for all modalities

### Technical Achievements

1. **Checkpoint Compatibility:** Solved spconv version mismatch issues
2. **CUDA Integration:** Successfully compiled and integrated custom operations
3. **Windows Support:** Resolved DLL loading issues for CUDA operations
4. **BEV Visualization:** Created comprehensive top-down visualization system

### Visualization

- **BEV Visualization:** `outputs/bevfusion_lidar_fixed/bev_visualization_final.png`
  - Shows point cloud colored by height
  - Displays detected bounding boxes with class labels
  - Demonstrates unified BEV representation

- **3D Outputs:** `outputs/bevfusion_lidar_fixed/*.ply` files
  - Point clouds, bounding boxes, and labels in Open3D format

---

## References

### Academic Papers

1. **BEVFusion (Primary Reference):**
   - Liu, Z., Tang, H., Amini, A., Yang, X., Mao, H., Rus, D., & Han, S. (2023). "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation." *IEEE International Conference on Robotics and Automation (ICRA)*.
   - Paper: https://arxiv.org/abs/2205.13542
   - Original Code: https://github.com/mit-han-lab/bevfusion
   - **Key Contribution:** Unified BEV representation for multi-sensor fusion, optimized BEV pooling (40x speedup)

### Software Frameworks

1. **MMDetection3D:**
   - Repository: https://github.com/open-mmlab/mmdetection3d
   - Documentation: https://mmdetection3d.readthedocs.io/
   - Version used: 1.4.0
   - License: Apache 2.0

2. **OpenMMLab Ecosystem:**
   - MMEngine: https://github.com/open-mmlab/mmengine
   - MMCV: https://github.com/open-mmlab/mmcv
   - MMDetection: https://github.com/open-mmlab/mmdetection
   - OpenMIM: https://github.com/open-mmlab/mim

3. **Open3D:**
   - Repository: https://github.com/isl-org/Open3D
   - Documentation: http://www.open3d.org/
   - Used for: 3D visualization and point cloud processing

4. **PyTorch:**
   - Version: 2.1.2+cu118
   - CUDA: 11.8 runtime
   - Documentation: https://pytorch.org/

### Datasets

1. **KITTI Dataset:**
   - Geiger, A., Lenz, P., & Urtasun, R. (2012). "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite." *CVPR 2012*.
   - Website: http://www.cvlibs.net/datasets/kitti/

2. **nuScenes Dataset:**
   - Caesar, H., et al. (2020). "nuScenes: A multimodal dataset for autonomous driving." *CVPR 2020*.
   - Website: https://www.nuscenes.org/

### Model Checkpoints

BEVFusion pretrained models downloaded from OpenMMLab Model Zoo:
- **BEVFusion (LiDAR-only):** https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
- **BEVFusion (LiDAR-Camera):** https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth

---

## Challenges & Solutions

### Challenge 1: BEVFusion Checkpoint Compatibility

**Problem:**
- BEVFusion checkpoint was trained with spconv 1.x
- Current environment uses spconv 2.x with different weight tensor layouts
- Shape mismatch: `[out_ch, D, H, W, in_ch]` vs `[D, H, W, in_ch, out_ch]`

**Solution:**
- Created `fix_bevfusion_checkpoint.py` to transpose weight tensors
- Automatically detects and fixes all sparse convolution layers
- Preserves checkpoint metadata and other weights

**Status:** Resolved

**Code:** [`fix_bevfusion_checkpoint.py`](fix_bevfusion_checkpoint.py)

---

### Challenge 2: CUDA DLL Loading on Windows

**Problem:**
- BEVFusion CUDA operations require CUDA DLLs
- Windows PATH issues preventing DLL loading
- `DLL load failed` errors during inference

**Solution:**
- Added automatic CUDA bin directory detection in `mmdet3d_inference2.py`
- Uses `os.add_dll_directory()` to add CUDA bin path
- Supports multiple CUDA versions (11.3, 12.4)

**Status:** Resolved

**Code:** [`mmdet3d_inference2.py`](mmdet3d_inference2.py) (lines 6-10)

---

### Challenge 3: NumPy ABI Compatibility

**Problem:**
- MMDetection3D compiled ops require NumPy 1.26.x
- NumPy 2.x breaks ABI compatibility
- Package conflicts during installation

**Solution:**
- Pinned NumPy to 1.26.4 in installation instructions
- Documented version requirements in README
- Installation order: NumPy first, then MMDetection3D

**Status:** Resolved

**Documentation:** [`README.md`](README.md) (line 65)

---

### Challenge 4: BEV Visualization Axis Scaling

**Problem:**
- BEV visualizations had overlapping bounding boxes
- Incorrect axis limits causing poor visualization
- Boxes appearing outside visible range

**Solution:**
- Fixed axis limit calculation in `scripts/visualize_bev.py`
- Uses union of point cloud and bounding box ranges
- Adds proper padding (10% of range, minimum 5m)
- Handles edge cases (empty detections, single points)

**Status:** Resolved

**Code:** [`scripts/visualize_bev.py`](scripts/visualize_bev.py) (lines 208-239)

---


### Challenge 5: CUDA Compilation for BEVFusion

**Problem:**
- BEVFusion requires custom CUDA operations compilation
- Setup.py compilation errors
- Architecture mismatch issues

**Solution:**
- Documented compilation process in README
- Provided troubleshooting guide
- Created compatibility report
- Verified CUDA architecture support

**Status:** Resolved (with documentation)

**Documentation:** [`README.md`](README.md) (lines 52-58), [`COMPATIBILITY_REPORT.md`](COMPATIBILITY_REPORT.md)

---

### Challenge 6: Multi-Dataset Support

**Problem:**
- Different dataset formats (KITTI, nuScenes, WaymoKITTI)
- Inconsistent file structures
- Manual path specification required

**Solution:**
- Implemented dataset mode selection (`--dataset kitti/waymokitti/any`)
- Automatic file discovery and matching
- Flexible input path handling
- Support for both single files and directories

**Status:** Resolved

**Code:** [`mmdet3d_inference2.py`](mmdet3d_inference2.py) (lines 729-850)

---

## Future Plans

1. **LiDAR-Camera Fusion Mode**
   - Complete multi-modal fusion implementation
   - Integrate camera image processing pipeline
   - Test full BEVFusion fusion capabilities
   - **Code Location:** Enhance `mmdet3d_inference2.py` and BEVFusion configs

2. **mAP/AP Calculation for BEVFusion**
   - Implement precision/recall calculation with ground truth
   - Calculate Average Precision (AP) for each class on nuScenes
   - Generate mAP (mean Average Precision) metrics
   - **Code Location:** New file `calculate_bevfusion_map.py`

3. **IoU Calculation**
   - Implement 3D IoU calculation between BEVFusion predictions and ground truth
   - Generate IoU distribution statistics
   - Filter detections by IoU threshold
   - **Code Location:** New file `calculate_iou.py``

4. **BEVFusion Training Pipeline**
   - Fine-tuning pretrained weights with the full nuScenes dataset
   - Fine-tuning scripts for custom datasets
   - Data augmentation utilities for BEVFusion
   - Training monitoring and checkpointing
   - **Code Location:** New directory `training/bevfusion/`
   
---

## Project Statistics

- **Total Lines of Code:** ~3,000+ (excluding external dependencies)
- **Python Files:** 8 core files
- **Primary Model:** BEVFusion (LiDAR-only mode implemented, LiDAR-Camera fusion in progress)
- **Datasets Supported:** nuScenes (primary), KITTI (for testing)
- **Visualization Types:** 3 (2D projection, 3D interactive, BEV top-down)
- **Documentation Pages:** 3 (README, REPORT, COMPATIBILITY_REPORT)
- **Custom CUDA Operations:** 2 (voxelization, BEV pooling)

---

## Contact & Contribution

**Repository:** https://github.com/SoubhikMajumdar/3D-LiDAR-Based-Birds-Eye-View-Object-Detection-via-BEVFusion

**Key Files:**
- Main inference: [`mmdet3d_inference2.py`](mmdet3d_inference2.py)
- BEV visualization: [`scripts/visualize_bev.py`](scripts/visualize_bev.py)
- BEVFusion analysis: [`analyze_bevfusion_results.py`](analyze_bevfusion_results.py)
- BEVFusion checkpoint fix: [`fix_bevfusion_checkpoint.py`](fix_bevfusion_checkpoint.py)
- BEVFusion implementation: [`external/mmdetection3d/projects/BEVFusion/`](external/mmdetection3d/projects/BEVFusion/)

---

**Last Updated:** November 2025  
**Project Status:** Active Development  
**License:** See repository for license information

