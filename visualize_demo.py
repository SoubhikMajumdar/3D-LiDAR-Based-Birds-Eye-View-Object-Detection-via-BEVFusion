
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.lines import Line2D

# NuScenes Class Mapping and Colors
CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Colors (Matplotlib compatible names or hex)
# car: blue, truck: orange, barrier: brown, pedestrian: yellow-green, traffic_cone: cyan
CLASS_COLORS = {
    0: 'tab:blue',      # car
    1: 'tab:orange',    # truck
    2: 'gold',          # construction_vehicle
    3: 'tab:red',       # bus
    4: 'darkorange',    # trailer
    5: 'sienna',        # barrier
    6: 'tab:purple',    # motorcycle
    7: 'tab:pink',      # bicycle
    8: 'tab:olive',     # pedestrian
    9: 'tab:cyan'       # traffic_cone
}

def get_corners(x, y, w, l, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[w/2, l/2], [w/2, -l/2], [-w/2, -l/2], [-w/2, l/2]]).T
    rotated_corners = np.dot(R, corners)
    corners_x = rotated_corners[0, :] + x
    corners_y = rotated_corners[1, :] + y
    return corners_x, corners_y

demo_dir = r"C:\Users\juand\OneDrive\Desktop\3D-LiDAR-Based-Birds-Eye-View-Object-Detection-via-BEVFusion\demo"
preds_dir = os.path.join(demo_dir, "outputs", "preds")
out_vis_dir = os.path.join(demo_dir, "outputs", "vis")
os.makedirs(out_vis_dir, exist_ok=True)

bin_files = glob.glob(os.path.join(demo_dir, "*.bin"))

print(f"Visualizing {len(bin_files)} files with enhanced style...")

for bin_file in bin_files:
    filename = os.path.basename(bin_file)
    json_name = filename.replace('.bin', '.json')
    json_path = os.path.join(preds_dir, json_name)
    
    if not os.path.exists(json_path):
        continue
        
    # Load Points
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
    
    # Load Predictions
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    bboxes = data['bboxes_3d']
    scores = data['scores_3d']
    labels = data['labels_3d']
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
    ax.set_facecolor('white')
    
    # Filter points
    pts_x = points[:, 0]
    pts_y = points[:, 1]
    pts_z = points[:, 2]
    mask = (pts_x > -54) & (pts_x < 54) & (pts_y > -54) & (pts_y < 54)
    
    # Plot Points colored by height (Z)
    sc = ax.scatter(pts_x[mask], pts_y[mask], c=pts_z[mask], s=0.5, cmap='viridis', alpha=0.3, vmin=-5, vmax=3)
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Height (Z)')
    
    # Track detected classes for legend
    detected_classes = set()
    detection_count = 0
    
    for i, bbox in enumerate(bboxes):
        score = scores[i]
        label_idx = labels[i]
        
        if score < 0.3: continue
        
        detection_count += 1
        detected_classes.add(label_idx)
        
        color = CLASS_COLORS.get(label_idx, 'black')
        
        # Draw Box
        x, y, z, dx, dy, dz, yaw = bbox[:7]
        cx, cy = get_corners(x, y, dx, dy, yaw)
        cx = np.append(cx, cx[0])
        cy = np.append(cy, cy[0])
        
        ax.plot(cx, cy, c=color, linewidth=1.5)
        
        # Add Label
        class_name = CLASSES[label_idx] if label_idx < len(CLASSES) else str(label_idx)
        ax.text(x, y, f'{class_name}\n{score:.2f}', color=color, fontsize=6, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

    # Legend
    legend_elements = [Line2D([0], [0], color=CLASS_COLORS[idx], lw=2, label=CLASSES[idx]) 
                       for idx in sorted(detected_classes)]
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Styling
    ax.set_title(f"BEV Visualization - {detection_count} detections (score $\geq$ 0.3)", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(-54, 54)
    ax.set_ylim(-54, 54)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal')
    
    out_name = filename + '.png'
    plt.savefig(os.path.join(out_vis_dir, out_name), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {out_name}")

print("Done visualizing.")
