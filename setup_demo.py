
import os
import shutil
import glob
import sys
sys.path.insert(0, os.path.abspath("external/mmdetection3d"))
from mmdet3d.apis import LidarDet3DInferencer

# Configuration
source_dir = r"C:\Users\juand\OneDrive\Desktop\3D-LiDAR-Based-Birds-Eye-View-Object-Detection-via-BEVFusion\data\v1.0-mini\samples\LIDAR_TOP"
demo_dir = r"C:\Users\juand\OneDrive\Desktop\3D-LiDAR-Based-Birds-Eye-View-Object-Detection-via-BEVFusion\demo"
config_file = os.path.abspath(r"external/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py")
checkpoint_file = os.path.abspath(r"work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/epoch_10.pth")
output_dir = os.path.join(demo_dir, "outputs")

# Ensure directories exist
os.makedirs(demo_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Get 10 sample files
all_files = sorted(glob.glob(os.path.join(source_dir, "*.bin")))
sample_files = all_files[:10]

print(f"Found {len(all_files)} files. Selecting first 10 for demo.")

copied_files = []
for src in sample_files:
    filename = os.path.basename(src)
    dst = os.path.join(demo_dir, filename)
    shutil.copy2(src, dst)
    copied_files.append(dst)
    print(f"Copied {filename}")

# Run Inference
print("Starting Inference...")
from mmengine.config import Config

print(f"Loading config from {config_file}")
cfg = Config.fromfile(config_file)
inferencer = LidarDet3DInferencer(model=cfg, weights=checkpoint_file, device='cuda:0')

for pcd_file in copied_files:
    print(f"Processing {pcd_file}...")
    inferencer(inputs=dict(points=pcd_file), out_dir=output_dir, show=False, pred_score_thr=0.3)

print(f"Done. Results saved to {output_dir}")
