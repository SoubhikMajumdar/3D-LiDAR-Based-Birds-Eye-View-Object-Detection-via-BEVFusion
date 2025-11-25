"""
Fix BEVFusion checkpoint shape mismatches.

The checkpoint was saved with spconv 1.x which uses weight layout:
[out_channels, D, H, W, in_channels]

But the current model expects spconv 2.x layout:
[D, H, W, in_channels, out_channels]

This script transposes the weights to match the expected layout.
"""

import torch
import argparse
from pathlib import Path


def transpose_spconv_weight(weight, from_layout='spconv1', to_layout='spconv2'):
    """
    Transpose sparse convolution weight from one layout to another.
    
    Args:
        weight: torch.Tensor with shape [out_ch, D, H, W, in_ch] (spconv1)
        from_layout: 'spconv1' or 'spconv2'
        to_layout: 'spconv1' or 'spconv2'
    
    Returns:
        Transposed weight tensor
    """
    if from_layout == 'spconv1' and to_layout == 'spconv2':
        # [out_ch, D, H, W, in_ch] -> [D, H, W, in_ch, out_ch]
        # Permute: (0, 1, 2, 3, 4) -> (1, 2, 3, 4, 0)
        if weight.dim() == 5:
            return weight.permute(1, 2, 3, 4, 0).contiguous()
    elif from_layout == 'spconv2' and to_layout == 'spconv1':
        # [D, H, W, in_ch, out_ch] -> [out_ch, D, H, W, in_ch]
        # Permute: (0, 1, 2, 3, 4) -> (4, 0, 1, 2, 3)
        if weight.dim() == 5:
            return weight.permute(4, 0, 1, 2, 3).contiguous()
    
    return weight


def fix_checkpoint(input_path, output_path):
    """
    Fix checkpoint by transposing spconv weights.
    
    Args:
        input_path: Path to original checkpoint
        output_path: Path to save fixed checkpoint
    """
    print(f"Loading checkpoint from {input_path}...")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    print(f"Found {len(state_dict)} parameters in checkpoint")
    
    # Find all mismatched layers (all in pts_middle_encoder)
    mismatched_keys = []
    for key in state_dict.keys():
        if 'pts_middle_encoder' in key and 'weight' in key:
            weight = state_dict[key]
            # Check if it's a 5D tensor (spconv weight)
            if weight.dim() == 5:
                mismatched_keys.append(key)
    
    print(f"Found {len(mismatched_keys)} spconv weight layers to fix:")
    for key in mismatched_keys:
        old_shape = state_dict[key].shape
        print(f"  {key}: {old_shape}")
    
    # Transpose the weights
    fixed_count = 0
    for key in mismatched_keys:
        old_weight = state_dict[key]
        old_shape = old_weight.shape
        
        # Transpose from spconv1 to spconv2 layout
        new_weight = transpose_spconv_weight(old_weight, 'spconv1', 'spconv2')
        new_shape = new_weight.shape
        
        print(f"  Fixed {key}: {old_shape} -> {new_shape}")
        state_dict[key] = new_weight
        fixed_count += 1
    
    # Update checkpoint
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = state_dict
    elif 'model' in checkpoint:
        checkpoint['model'] = state_dict
    else:
        checkpoint = state_dict
    
    # Save fixed checkpoint
    print(f"\nSaving fixed checkpoint to {output_path}...")
    torch.save(checkpoint, output_path)
    print(f"✅ Fixed {fixed_count} weight tensors")
    print(f"✅ Checkpoint saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fix BEVFusion checkpoint shape mismatches')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input checkpoint file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output fixed checkpoint (default: input_path_fixed.pth)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        exit(1)
    
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"
    else:
        output_path = Path(args.output)
    
    fix_checkpoint(input_path, output_path)

