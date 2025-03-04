import sys
import os
sys.path.append('/projects/VLN-Tutorial/seq2seq')
import cv2
import numpy as np
from env import EnvBatch, R2RBatch

def visualize_obs(obs, save_dir):
    rgb = obs['rgb']
    depth = obs['depth']
    
    # Save RGB image
    cv2.imwrite(f'{save_dir}/rgb.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    # Normalize depth (uint16) to uint8 range for visualization
    depth = depth.astype(float)  # Convert to float for calculation
    depth_normalized = ((depth - depth.min()) * 255 / (depth.max() - depth.min())).astype('uint8')
    # Remove the extra dimension if shape is (480, 640, 1)
    depth_normalized = depth_normalized.squeeze()
    cv2.imwrite(f'{save_dir}/depth.png', depth_normalized)
    print(f"RGB shape: {rgb.shape}, saved to {save_dir}/rgb.png")
    print(f"Depth shape: {depth.shape}, saved to {save_dir}/depth.png")

env = R2RBatch(feature_store=False, enable_depth=True, batch_size=1)

obs = env.reset()
save_dir = 'tmp'
os.makedirs(save_dir, exist_ok=True)
visualize_obs(obs[0], save_dir)