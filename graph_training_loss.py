import re
import matplotlib.pyplot as plt
import os

log_file = 'work_dirs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d/20251214_225217/20251214_225217.log'
output_loss_file = 'training_loss_rerun.png'
output_map_file = 'validation_map_rerun.png'

losses = []
iterations = []
val_epochs = []
val_maps = []

# Regex patterns
# Loss: Epoch(train)  [5][ 50/408] ... loss: 1.0279 ...
loss_pattern = re.compile(r'Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/(\d+)\] .* loss: ([\d\.]+)')
# mAP: matches "NuScenes metric/pred_instances_3d_NuScenes/mAP: 0.3923"
map_pattern = re.compile(r'NuScenes metric/pred_instances_3d_NuScenes/mAP:\s+([\d\.]+)')
# Epoch val line to associate with map
val_pattern = re.compile(r'Epoch\(val\)\s+\[(\d+)\]\[\d+/\d+\].*NuScenes metric/pred_instances_3d_NuScenes/mAP:\s+([\d\.]+)')

if not os.path.exists(log_file):
    print(f"Error: {log_file} not found.")
    exit(1)

print(f"Reading {log_file}...")
with open(log_file, 'r') as f:
    for line in f:
        # Loss
        loss_match = loss_pattern.search(line)
        if loss_match:
            epoch = int(loss_match.group(1))
            iter_step = int(loss_match.group(2))
            total_iters = int(loss_match.group(3))
            loss = float(loss_match.group(4))
            global_step = (epoch - 1) * total_iters + iter_step
            losses.append(loss)
            iterations.append(global_step)
        
        # Validation mAP
        val_match = val_pattern.search(line)
        if val_match:
            epoch = int(val_match.group(1))
            map_score = float(val_match.group(2))
            val_epochs.append(epoch)
            val_maps.append(map_score)

print(f"Found {len(losses)} loss points.")
print(f"Found {len(val_maps)} mAP points: {val_maps} at epochs {val_epochs}")

# Plot Loss
if losses:
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, losses, label='Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss (10 Epochs)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_loss_file)
    print(f"Saved loss plot to {output_loss_file}")

# Plot mAP
if val_maps:
    plt.figure(figsize=(10, 6))
    plt.plot(val_epochs, val_maps, marker='o', linestyle='-', color='orange', label='Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Validation mAP over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_map_file)
    print(f"Saved mAP plot to {output_map_file}")
