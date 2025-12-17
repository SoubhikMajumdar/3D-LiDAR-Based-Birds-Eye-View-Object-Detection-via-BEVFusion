# Final Project Report: BEVFusion Implementation

## Introduction
BEVFusion represents a state-of-the-art approach to 3D object detection that leverages Bird's-Eye View (BEV) representations to fuse multi-modal sensor data from LiDAR and camera sensors into a unified spatial framework. This report documents the technical implementation of BEVFusion's LiDAR-only variant, detailing the model architecture, pretrained weight integration, and the systematic development process that enabled successful inference and visualization capabilities. The implementation is built upon the OpenMMLab MMDetection3D framework, which provides the foundational infrastructure for 3D detection models, while BEVFusion extends this framework with specialized BEV-based fusion mechanisms.

## Model Architecture
The BEVFusion LiDAR-only architecture consists of five interconnected components that transform raw point cloud data into 3D object detections. The pipeline begins with the voxel encoder, which processes raw LiDAR points containing five-dimensional features (x, y, z coordinates, intensity, and timestamp) and converts them into a structured voxel representation. This voxelization process discretizes the continuous 3D space into a sparse grid with dimensions of 1440×1440×41 voxels, covering a spatial range of 108 meters in the X and Y directions and 8 meters in height, with each voxel measuring 0.075×0.075×0.2 meters.

The sparse middle encoder, implemented as a BEVFusionSparseEncoder, processes these voxelized features through a four-stage sparse 3D convolutional neural network. The architecture progressively increases feature channel depth from the initial 5 input channels through stages that expand to 16, 32, 64, and finally 128 channels, ultimately producing 256-channel sparse feature maps. This component leverages sparse convolution operations (spconv) to achieve computational efficiency by processing only non-empty voxels, which typically represent less than 5% of the total voxel grid. The efficiency gains from sparse processing are substantial, providing 10-100x speedup compared to dense 3D convolutions while maintaining equivalent representational capacity.

The backbone network, based on the SECOND (Sparsely Embedded Convolutional Detection) architecture, operates on the 256-channel sparse features extracted by the middle encoder. This component applies 2D convolutions to the BEV plane, treating the height dimension as feature channels, and produces multi-scale feature representations with 128 and 256 channels at different spatial resolutions. The backbone employs two stages with stride configurations of 1 and 2, enabling the network to capture both fine-grained local features and broader contextual information necessary for detecting objects at various scales.

The neck component, implemented as a SECONDFPN (Feature Pyramid Network), fuses the multi-scale features from the backbone through upsampling and concatenation operations. This fusion produces unified feature maps with 256 channels at each scale, which are then concatenated to form a 512-channel feature representation that serves as input to the detection head. The FPN architecture ensures that both high-resolution fine details and low-resolution semantic information are preserved and effectively combined.

The detection head employs a TransFusionHead architecture, which is a transformer-based decoder that generates object proposals through learnable query mechanisms. The head maintains 200 object queries that interact with the BEV feature maps through cross-attention mechanisms, enabling the model to attend to relevant spatial regions for object detection. The transformer decoder consists of a single layer with 8 attention heads operating on 128-dimensional hidden representations. The head outputs 3D bounding boxes for 10 object classes defined in the nuScenes dataset: car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, and traffic_cone. Training employs a combination of Focal Loss for classification, Gaussian Focal Loss for heatmap regression, and L1 Loss for bounding box regression, with respective weights of 1.0, 1.0, and 0.25.

## Theoretical Foundations

### Bird's-Eye View Representation
The fundamental theoretical innovation of BEVFusion lies in its use of Bird's-Eye View representations as a unified spatial framework for multi-modal sensor fusion. BEV representations project 3D spatial information onto a 2D plane viewed from above, creating a top-down perspective that naturally aligns with human spatial understanding and autonomous vehicle planning systems. This representation offers several theoretical advantages: first, it provides a consistent coordinate system that eliminates perspective distortion inherent in camera-based views; second, it enables direct spatial reasoning about object locations and relationships without complex coordinate transformations; third, it facilitates the fusion of heterogeneous sensor modalities (LiDAR point clouds and camera images) into a common feature space.

From a geometric perspective, the BEV transformation involves projecting 3D points onto a horizontal plane by discarding the vertical (Z) dimension or encoding it as a feature channel. This projection creates a 2D grid where each cell represents a spatial region in the horizontal plane, with the vertical information preserved as feature channels. The voxelization process quantizes continuous 3D space into discrete grid cells, enabling the application of standard 2D convolutional operations while maintaining 3D spatial awareness through the feature channel dimension.

### Sparse Convolution Theory
Sparse convolutions represent a fundamental algorithmic innovation that exploits the inherent sparsity of 3D point cloud data. Traditional dense convolutions operate on all voxels in a grid, regardless of whether they contain data, leading to computational waste when processing sparse point clouds where typically less than 5% of voxels are non-empty. Sparse convolutions theoretically reduce computational complexity from O(N) to O(S), where N is the total number of voxels and S is the number of non-empty voxels, providing substantial efficiency gains without sacrificing representational capacity.

The mathematical foundation of sparse convolutions lies in the observation that convolution operations can be reformulated to operate only on active (non-empty) voxels and their neighbors. This reformulation requires maintaining an active index list and computing convolution outputs only for positions where input features exist. The spconv library implements this through hash-based indexing and rule-based convolution kernels that dynamically determine which output positions to compute based on the sparsity pattern of the input. This approach maintains the same mathematical properties as dense convolutions (linearity, translation equivariance) while achieving orders of magnitude speedup for sparse data.

### Transformer Attention Mechanisms
The TransFusionHead employs transformer-based attention mechanisms to enable the model to dynamically attend to relevant spatial regions in the BEV feature map. The theoretical foundation of this approach lies in the attention mechanism's ability to learn content-adaptive feature aggregation. Unlike fixed receptive fields in convolutional networks, attention mechanisms allow each object query to attend to arbitrary spatial locations in the BEV feature map based on learned attention weights.

The cross-attention mechanism in the transformer decoder computes attention scores between each of the 200 learnable object queries and all positions in the BEV feature map. Mathematically, this can be expressed as Attention(Q, K, V) = softmax(QK^T / √d_k)V, where Q represents the query embeddings, K and V represent the key and value projections of the BEV features, and d_k is the dimension of the key vectors. The learned attention weights enable the model to focus on spatially relevant regions for each object query, allowing the detection head to adaptively extract features from different parts of the scene based on the query's learned representation.

The use of 200 fixed object queries represents a design choice that balances computational efficiency with detection capacity. Each query learns to specialize in detecting objects at specific spatial locations or with specific characteristics. The Hungarian matching algorithm used during training assigns ground truth objects to queries, enabling the model to learn query-specific detection patterns. This approach differs from anchor-based detection methods by using learnable queries rather than predefined anchor boxes, providing greater flexibility in object localization.

## Data Processing Pipeline
The data processing pipeline for BEVFusion inference involves several critical transformations that prepare raw LiDAR data for model input. The pipeline begins with loading point cloud data from binary files in the nuScenes format, which contains 5D point features. The system then applies multi-sweep fusion, loading the current LiDAR frame along with 9 historical sweeps to provide temporal context. This temporal aggregation significantly improves detection performance, particularly for moving objects, by providing multiple observations of the same scene from slightly different viewpoints and time instances.

The point cloud data undergoes range filtering to remove points outside the model's detection range of [-54, -54, -5, 54, 54, 3] meters, which corresponds to the voxel grid dimensions. Points are then voxelized according to the specified voxel size of [0.075, 0.075, 0.2] meters, with a maximum of 120,000 voxels during training and 160,000 during inference. The voxelization process groups points within each voxel and applies feature encoding to produce per-voxel feature vectors.

## Fine-Tuning
The fine-tuning process involved adapting the pre-trained BEVFusion model to a specific subset of the nuScenes dataset to evaluate performance improvements and domain adaptation capabilities. This section details the systematic approach taken, the challenges encountered, and the final results obtained.

### Methodology and Journey

The fine-tuning process was iterative, involving several key phases and technical problem-solving steps:

1.  **Data Preparation & Path Resolution**:
    *   **Challenge**: The initial setup faced "Path Duplication" errors (e.g., `samples/LIDAR_TOP/samples/LIDAR_TOP`) due to hardcoded expectations in the NuScenes SDK and our directory structure.
    *   **Solution**: We employed **Filesystem Junctions** to map `data/nuscenes` to our dataset source (`data/v1.0-mini` or `data/nuscenes-full`). This allowed the training scripts to see the expected directory structure without physically duplicating terabytes of data.

2.  **Dataset Sanitization (Full Dataset)**:
    *   **Challenge**: We utilized a partial download of the full nuScenes dataset ("Blob 01") alongside the full metadata. This caused crashes because the training loader attempted to access sweep files referenced in the metadata that were not present on disk.
    *   **Solution**: We developed a custom sanitization script (`sanitize_pkl.py`) that verified the physical existence of every referenced LiDAR file. It removed approximately 10% of the sweep references from the `.pkl` data info files, ensuring the dataloader only requested valid files.

3.  **Validation Logic Repair**:
    *   **Challenge**: The built-in validation step failed during training because the official evaluation script expected predictions for *all* samples in the validation set, whereas we were only running on the available subset.
    *   **Solution**: We "monkeypatched" the `NuScenesEval` class in `nuscenes_metric.py` to dynamically load ground truth annotations only for the samples that were actually predicted. This allowed us to compute valid mAP and NDS metrics for our partial dataset without the process crashing.

### Instructions for Fine-Tuning

To replicate this fine-tuning process on a new machine:

1.  **Prepare Data**: Ensure your nuScenes data is organized in `data/nuscenes`. If using a partial dataset, run `sanitize_pkl.py` to clean the `.pkl` info files.
2.  **Configure Training**:
    *   File: `projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`
    *   **Load Checkpoint**: Set `load_from` to the base `bevfusion_lidar_..._fixed.pth`.
    *   **Learning Rate**: Set `lr = 1e-4` (Standard for fine-tuning this model).
    *   **Epochs**: Set `max_epochs = 10`.
    *   **Scheduler**: Ensure the `CosineAnnealingLR` is configured for the active epoch range (0-10).
3.  **Launch Training**:
    ```bash
    python tools/train.py projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
    ```

### Final Results

After resolving the data and configuration issues, we successfully fine-tuned the model for 10 epochs.

**Performance Metrics:**

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **mAP (Mean Average Precision)** | **0.5430 - 0.5869** | The model achieved a peak accuracy of ~58.7% in initial tests, stabilizing around 54.3% in the re-run. This represents strong detection capability. |
| **NDS (NuScenes Detection Score)** | **0.6050 - 0.6461** | The NDS, which accounts for velocity and attribute estimation, remained high (>60%), indicating the model is robust in tracking object states. |

**Per-Class Performance (Car & Pedestrian):**
The model excelled at detecting common dynamic objects:
*   **Car**: ~85% AP
*   **Pedestrian**: ~84% AP

### Visualizations

**Training Loss vs. Epochs:**
The training loss demonstrated a steady, healthy convergence, confirming that the model effectively learned from the dataset without diverging.

![Training Loss Curve](file:///C:/Users/juand/OneDrive/Desktop/3D-LiDAR-Based-Birds-Eye-View-Object-Detection-via-BEVFusion/training_loss_rerun.png)

**Validation Accuracy (mAP) vs. Epochs:**
The validation accuracy started high (benefiting from pre-trained weights) and remained stable throughout the process.

![Validation mAP Curve](file:///C:/Users/juand/OneDrive/Desktop/3D-LiDAR-Based-Birds-Eye-View-Object-Detection-via-BEVFusion/validation_map_rerun.png)
