"""
Generate Bird's-Eye View (BEV) visualization of 3D object detection predictions.
Shows point cloud from top-down view with predicted bounding boxes.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def load_point_cloud(file_path):
    """Load point cloud from .pcd, .ply, or .bin file."""
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == '.bin':
        # KITTI-style binary format
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points[:, :3]  # Return x, y, z
    elif ext in ['.pcd', '.ply']:
        # Try to load with open3d first
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(file_path)
            return np.asarray(pcd.points)
        except ImportError:
            # Fallback: parse PLY manually
            return load_ply_manual(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def load_ply_manual(file_path):
    """Manually parse PLY file to extract points."""
    points = []
    with open(file_path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if b'end_header' in line:
                break
        
        # Check if binary or ASCII
        is_binary = b'binary' in b''.join(header_lines[:3])
        
        if is_binary:
            # Binary PLY - read as little-endian float32
            # This is a simplified parser - assumes standard format
            data = np.fromfile(f, dtype=np.float32)
            # Assume 3 floats per point (x, y, z)
            n_points = len(data) // 3
            points = data[:n_points * 3].reshape(n_points, 3)
        else:
            # ASCII PLY
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                    except ValueError:
                        continue
    
    return np.array(points)


def load_predictions(json_path):
    """Load predictions from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def draw_bev_box(ax, bbox, color='red', linewidth=2, alpha=0.8, label=None):
    """
    Draw a 3D bounding box in BEV (top-down view).
    
    Args:
        ax: matplotlib axis
        bbox: [x, y, z, w, l, h, yaw, ...] in LiDAR coordinates (may include velocity)
        color: box color
        linewidth: line width
        alpha: transparency
        label: optional text label
    """
    # Handle both 7-element and 9-element bboxes (9-element includes velocity)
    if len(bbox) >= 7:
        x, y, z, x_size, y_size, z_size, yaw = bbox[:7]
    else:
        raise ValueError(f"Bbox must have at least 7 elements, got {len(bbox)}")
    
    # For mmdet3d LiDAR boxes: format is [x, y, z, x_size, y_size, z_size, yaw]
    # According to mmdet3d documentation:
    # - x_size = size along X-axis (forward direction) = LENGTH
    # - y_size = size along Y-axis (left direction) = WIDTH  
    # - z_size = size along Z-axis (up direction) = HEIGHT
    # So: [x, y, z, length, width, height, yaw]
    l = x_size  # length (forward, X-axis)
    w = y_size  # width (left-right, Y-axis)
    h = z_size  # height (up, Z-axis)
    
    # Create box corners in local frame (centered at origin)
    # In BEV (top-down): X is forward, Y is left
    # Length is along X-axis, Width is along Y-axis
    corners_local = np.array([
        [-l/2, -w/2],  # back-left
        [l/2, -w/2],   # front-left
        [l/2, w/2],    # front-right
        [-l/2, w/2],   # back-right
    ])
    
    # Rotate by yaw
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([[cos_yaw, -sin_yaw],
                  [sin_yaw, cos_yaw]])
    corners_rotated = corners_local @ R.T
    
    # Translate to box center
    corners_rotated[:, 0] += x
    corners_rotated[:, 1] += y
    
    # Close the box
    corners_rotated = np.vstack([corners_rotated, corners_rotated[0]])
    
    # Draw box
    ax.plot(corners_rotated[:, 0], corners_rotated[:, 1], 
            color=color, linewidth=linewidth, alpha=alpha)
    
    # Draw front direction (arrow)
    front_vec = np.array([l/2 * cos_yaw, l/2 * sin_yaw])
    ax.arrow(x, y, front_vec[0], front_vec[1], 
             head_width=0.5, head_length=0.5, fc=color, ec=color, alpha=alpha)
    
    # Add label if provided
    if label:
        ax.text(x, y, label, fontsize=8, color=color, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


def visualize_bev(points, predictions, output_path, score_threshold=0.1, 
                  point_size=0.5, figsize=(12, 12)):
    """
    Create BEV visualization.
    
    Args:
        points: (N, 3) point cloud
        predictions: dict with 'bboxes_3d', 'scores_3d', 'labels_3d'
        output_path: path to save the image
        score_threshold: minimum score to display
        point_size: size of points in scatter plot
        figsize: figure size
    """
    bboxes = np.array(predictions['bboxes_3d'])
    scores = np.array(predictions['scores_3d'])
    labels = np.array(predictions['labels_3d'])
    
    # Filter by score threshold
    mask = scores >= score_threshold
    bboxes = bboxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Class names for nuScenes
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    
    # Color map for classes
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    class_colors = {i: colors[i] for i in range(len(class_names))}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot point cloud (top-down view: x vs y)
    # Color points by height (z)
    z_values = points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    if z_max > z_min:
        z_norm = (z_values - z_min) / (z_max - z_min)
    else:
        z_norm = np.zeros_like(z_values)
    scatter = ax.scatter(points[:, 0], points[:, 1], c=z_norm, 
                        cmap='viridis', s=point_size, alpha=0.3, edgecolors='none')
    
    # Draw bounding boxes
    for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
        cls_name = class_names[label] if label < len(class_names) else f'class_{label}'
        color = class_colors.get(label, 'red')
        
        # Create label text
        label_text = f'{cls_name}\n{score:.2f}'
        
        # Draw box
        draw_bev_box(ax, bbox, color=color, label=label_text if score > 0.3 else None)
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'BEV Visualization - {len(bboxes)} detections (score ≥ {score_threshold})', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set axis limits based on data ranges to ensure proper scaling
    if len(bboxes) > 0:
        x_min, x_max = bboxes[:, 0].min(), bboxes[:, 0].max()
        y_min, y_max = bboxes[:, 1].min(), bboxes[:, 1].max()
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = max(x_range * 0.1, 5.0)
        y_pad = max(y_range * 0.1, 5.0)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    # Also set limits based on point cloud if available
    if len(points) > 0:
        pc_x_min, pc_x_max = points[:, 0].min(), points[:, 0].max()
        pc_y_min, pc_y_max = points[:, 1].min(), points[:, 1].max()
        if len(bboxes) > 0:
            # Use union of bbox and point cloud ranges
            x_min = min(bboxes[:, 0].min(), pc_x_min)
            x_max = max(bboxes[:, 0].max(), pc_x_max)
            y_min = min(bboxes[:, 1].min(), pc_y_min)
            y_max = max(bboxes[:, 1].max(), pc_y_max)
        else:
            x_min, x_max = pc_x_min, pc_x_max
            y_min, y_max = pc_y_min, pc_y_max
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = max(x_range * 0.1, 5.0)
        y_pad = max(y_range * 0.1, 5.0)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    # Add legend
    legend_elements = []
    for i, name in enumerate(class_names):
        if i in labels:
            legend_elements.append(patches.Patch(color=class_colors[i], label=name))
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Add colorbar for height
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Height (Z)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"BEV visualization saved to: {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate BEV visualization from predictions')
    parser.add_argument('--points', type=str, required=True,
                        help='Path to point cloud file (.pcd, .ply, or .bin)')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSON file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output image path (default: predictions_dir/bev_vis.png)')
    parser.add_argument('--score-thr', type=float, default=0.1,
                        help='Score threshold for displaying boxes (default: 0.1)')
    parser.add_argument('--point-size', type=float, default=0.5,
                        help='Point size in scatter plot (default: 0.5)')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading point cloud from: {args.points}")
    points = load_point_cloud(args.points)
    print(f"  Loaded {len(points)} points")
    
    print(f"Loading predictions from: {args.predictions}")
    predictions = load_predictions(args.predictions)
    print(f"  Loaded {len(predictions['bboxes_3d'])} predictions")
    
    # Determine output path
    if args.output is None:
        pred_dir = os.path.dirname(args.predictions)
        args.output = os.path.join(pred_dir, 'bev_visualization.png')
    
    # Generate visualization
    visualize_bev(points, predictions, args.output, 
                  score_threshold=args.score_thr,
                  point_size=args.point_size)
    
    print(f"\nDone! BEV visualization saved to: {args.output}")


if __name__ == '__main__':
    main()

