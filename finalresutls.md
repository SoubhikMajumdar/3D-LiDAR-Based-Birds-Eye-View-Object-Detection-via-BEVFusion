# Final Experiment Results

This document summarizes the results of various fine-tuning strategies on the nuScenes dataset.

## 1. Comparing Fine-Tuning Strategies (Mini Dataset)

We compared the original pre-trained baseline and varying fine-tuning strategies on the `v1.0-mini` dataset.

| Metric | Baseline (Pre-trained) | 5 Epochs (LR 1e-5) | 30 Epochs (LR 5e-5) | **12 Epochs (LR 2e-6)** | Frozen Backbone | **Strong Aug (LR 1e-5)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NDS** | **0.5690** | **0.5472** | **0.4671** | **0.5544** | 0.5241 | 0.5293 |
| **mAP** | **0.5512** | **0.5101** | **0.3836** | **0.5225** | 0.4691 | 0.4819 |

### Visualizations

**Conservative Run (12 Epochs, Learning Rate 2e-6):**
The most stable strategy, maintaining high performance without catastrophic forgetting (overfitting).

*   **Training Loss:**
    ![Training Loss Curve (12e)](file:///C:/Users/juand/.gemini/antigravity/brain/cc8e3c88-978c-4720-bdba-5a993e951900/training_loss_12e.png)

*   **Validation mAP:**
    ![Validation mAP Curve (12e)](file:///C:/Users/juand/.gemini/antigravity/brain/cc8e3c88-978c-4720-bdba-5a993e951900/validation_map_12e.png)

---

## 2. Final Fine-Tuning (Partial Full Dataset)

Training on the sanitized `v1.0-trainval01` subset (2462 samples).

| Metric | Score | Note |
| :--- | :--- | :--- |
| **NDS** | **0.6050** | Validated re-run (with graphs). |
| **mAP** | **0.5430** | Validated re-run (with graphs). |

### Per-Class Performance
- **Car**: 0.849 AP
- **Pedestrian**: 0.842 AP
- **Bus**: 0.686 AP
- **Truck**: 0.555 AP

### Visualizations

*   **Training Loss:**
    ![Training Loss Curve (Re-run)](file:///C:/Users/juand/OneDrive/Desktop/3D-LiDAR-Based-Birds-Eye-View-Object-Detection-via-BEVFusion/training_loss_rerun.png)

*   **Validation Accuracy (mAP):**
    ![Validation mAP Curve (Re-run)](file:///C:/Users/juand/OneDrive/Desktop/3D-LiDAR-Based-Birds-Eye-View-Object-Detection-via-BEVFusion/validation_map_rerun.png)
