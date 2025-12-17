# BEVFusion: LiDAR-Based 3D Object Detection (Fine-Tuning on nuScenes)

This repository contains a **customized and fine-tuned implementation of BEVFusion** for 3D object detection using LiDAR data from the **nuScenes dataset**.

The project focuses on:
1.  **Fine-tuning** the model on a partial dataset ("blob01").
2.  **Customizing** the inference pipeline for demo generation.
3.  **Visualizing** results with high-quality BEV plots (Z-colored points, class-specific boxes).

---

## 📸 Demo Results

We successfully fine-tuned the model and ran inference on validation samples.

### Bird's Eye View (BEV) Visualization
*Point cloud colored by height (Z), detections in class-specific colors.*

![Demo Preview](demo_preview_v2.png)

---

## 🛠️ Setup & Installation

### 1. Environment
**OS**: Windows (tested) or Linux (WSL2)
**Python**: 3.8+
**CUDA**: 11.3+ (Tested with 11.8/12.x on RTX GPU)

### 2. Install Dependencies
This project relies on `mmdetection3d` and its ecosystem.

```bash
# 1. Install PyTorch (ensure CUDA match)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 2. Install MMEngine, MMCV, MMDetection
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

# 3. Install MMDetection3D (from source/submodule)
cd external/mmdetection3d
pip install -v -e .
cd ../..
```

### 3. Compile BEVFusion
The BEVFusion ops must be compiled.
```bash
cd external/mmdetection3d/projects/BEVFusion
python setup.py develop
cd ../../../..
```

---

## 📊 Models & Datasets

### Dataset: nuScenes (v1.0-mini & v1.0-trainval partial)
We focused on fine-tuning using a **partial slice** of the full nuScenes dataset (referred to as `blob01`), containing ~2500 training samples.

*   **Challenge**: The metadata contained citations for files not present in the partial download.
*   **Solution**: We developed `sanitize_pkl.py` to filter the dataset info files, removing disjoint references to prevent training crashes.

### Model: BEVFusion (LiDAR Only)
*   **Backbone**: VoxelNet
*   **Neck**: SECONDFPN
*   **Head**: TransFusionHead
*   **Config**: `bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`

---

## 📈 Experiment Results (Fine-Tuning)

We compared several strategies to adapt the pre-trained model to our specific data subset.

| Strategy | Learning Rate | Epochs | mAP | NDS | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | N/A | 0 | **0.551** | **0.569** | Original Pre-trained |
| **Fail** | 5e-5 | 30 | 0.384 | 0.467 | Catastrophic Forgetting |
| **Gentle** | 2e-6 | 12 | **0.523** | **0.554** | Best Stability |
| **Full Run** | 1e-4 | 10 | **0.543** | **0.605** | **Final Selected Model** |

### Training Loss vs Epochs (Final Run)
*Loss decreased steadily, showing effective learning despite the partial dataset.*
![Training Loss](training_loss_rerun.png)

### Validation Accuracy (mAP) vs Epochs
*Accuracy remained high, validating the "sanitized" dataset approach.*
![Validation mAP](validation_map_rerun.png)

---

## 🚀 How to Run

### 1. Fine-Tuning
To reproduce the training:

1.  **Sanitize Data**: `python sanitize_pkl.py`
2.  **Run Training**:
    ```powershell
    $env:PYTHONPATH = "external\mmdetection3d;$env:PYTHONPATH"
    python external\mmdetection3d\tools\train.py external\mmdetection3d\projects\BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
    ```

### 2. Demo generation
To run inference on sample files and generate visualizations:

```bash
# 1. Setup Demo (Inference)
python setup_demo.py
# This creates demo/outputs/preds/*.json

# 2. Visualize (Generate Images)
python visualize_demo.py
# Output: demo/outputs/vis/*.png
```

---

## ⚠️ Takeaways & Limitations

1.  **Partial Data Complexity**: Training on a dataset slice is non-trivial. Standard tools assume the full dataset exists. **Sanitization** of metadata is critical to prevent "File Not Found" crashes during evaluation.
2.  **Submodule Management**: BEVFusion is often implemented as a plugin. Managing python paths (`PYTHONPATH`) and checking ignored submodule files is essential for reproducibility.
3.  **Hyperparameters matter**: A high learning rate (e.g., 5e-5) on a small/partial dataset leads to rapid **catastrophic forgetting**. A lower, "gentle" learning rate (2e-6) preserved pre-trained knowledge much better.
4.  **Windows Compilation**: Compiling custom CUDA ops (GridSampler, Voxelization) on Windows requires careful setup of `DISTUTILS_USE_SDK=1` and Visual Studio environments.

---

**Authors**: Juan D. Liang
**Remote**: [SoubhikMajumdar/3D-LiDAR-Based-Birds-Eye-View-Object-Detection-via-BEVFusion](https://github.com/SoubhikMajumdar/3D-LiDAR-Based-Birds-Eye-View-Object-Detection-via-BEVFusion)
