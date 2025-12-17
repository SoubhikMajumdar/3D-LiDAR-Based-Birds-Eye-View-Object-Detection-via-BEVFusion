# How to Fine-Tune BEVFusion on `blob01` (Partial NuScenes)

This guide details the steps to start a training run from scratch using **only** the data contained in `v1.0-trainval01_blobs.tgz` (referred to as "blob1").

Since "blob1" is only a slice of the full dataset, the standard metadata files (`.pkl`) will contain references to missing files. Attempting to train directly will cause crashes. We must **sanitize** the dataset first.

## Prerequisites
- You have `data/v1.0-trainval01_blobs.tgz`.
- You have the metadata `v1.0-trainval_meta.tgz` (or the unpacked JSONs in `data/nuscenes/v1.0-trainval`).
- You have my custom script `sanitize_pkl.py` (already in the root folder).

---

## Step 1: Data Preparation

1.  **Extract the Data**
    Unzip the blob and the metadata into the standard `data/nuscenes` structure.
    ```bash
    # Conceptually:
    # Extract 'v1.0-trainval01_blobs' contents -> data/nuscenes/samples, sweeps, etc.
    # Extract 'v1.0-trainval_meta' contents -> data/nuscenes/v1.0-trainval/
    ```

    *Note: Ensure `data/nuscenes` contains `samples/`, `sweeps/`, `maps/` and `v1.0-trainval/`.*

2.  **Generate Standard Infos (if not already done)**
    Run the MMDetection3D tool to create the initial `.pkl` files (this covers the *entire* dataset, valid or not).
    ```bash
    python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
    ```
    *Output:* `data/nuscenes/nuscenes_infos_train.pkl`, `...val.pkl`.

3.  **Sanitize the Infos (Crucial Step)**
    This step removes entries from the `.pkl` files that correspond to missing sensor data (i.e., data not in blob1).
    ```bash
    python sanitize_pkl.py
    ```
    *Output:* `data/nuscenes/nuscenes_infos_train_sanitized.pkl`, `...val_sanitized.pkl`.

---

## Step 2: Configuration Changes

Modify the config file: `external/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`.

### 1. Update Data Paths
Point the dataset to your **sanitized** info files.

```python
# In the config file:

# 1. Update data_root if needed (usually just 'data/nuscenes/')
data_root = 'data/nuscenes/'

# 2. Update the annotation files to use the SANITIZED versions
train_dataloader = dict(
    dataset=dict(
        # ...
        ann_file='nuscenes_infos_train_sanitized.pkl',  # <--- CHANGED
        # ...
    )
)

val_dataloader = dict(
    dataset=dict(
        # ...
        ann_file='nuscenes_infos_val_sanitized.pkl',    # <--- CHANGED
        # ...
    )
)
```

### 2. Adjust Training Parameters
For a "fine-tune" (starting from a pre-trained model but training on new data):

```python
# 1. Load the pre-trained base (NOT your previous epoch_10.pth, but the original checkpoint)
load_from = 'checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933_fixed.pth'

# 2. Set Learning Rate (Lower for fine-tuning, standard for full train)
# If fine-tuning: 1e-4 or 2e-5. If Full Training: 1e-4.
optim_wrapper = dict(
    optimizer=dict(lr=0.0001)
)

# 3. Epochs
train_cfg = dict(
    by_epoch=True, 
    max_epochs=20,     # Set how long you want to train
    val_interval=1
)
```

---

## Step 3: Run Training

Execute the training command.

```bash
python tools/train.py external/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
```

## Summary of Files to Change
1.  **`sanitize_pkl.py`**: Run this once.
2.  **`projects/.../bevfusion_..._nus-3d.py`**:
    - Change `ann_file` to `*_sanitized.pkl`.
    - Set `load_from` to the base checkpoint.
    - Adjust `lr` and `max_epochs`.


# 1. Add the directory to PYTHONPATH (PowerShell syntax)
$env:PYTHONPATH = "external\mmdetection3d;$env:PYTHONPATH"
# 2. Run the training script
python external\mmdetection3d\tools\train.py external\mmdetection3d\projects\BEVFusion\configs\bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py