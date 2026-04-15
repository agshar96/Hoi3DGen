import numpy as np
import cv2
import trimesh
import json
import torch
import os

def pc_from_depth(depth_file_path, rgb_file_path, camera_fov, cam_transform, segmentation_mask=None, erode_mask=False):
    """
    Reproject depth map to 3D point cloud and save as trimesh point cloud.
    
    Args:
        depth_file_path: Path to depth map PNG file
        rgb_file_path: Path to RGB image file
        camera_fov: Camera field of view in degrees
    """
    # Load depth map and RGB image
    depth_img = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    depth_max = 2.866
    depth_min = 1.134

    # Convert back to meters
    depth_meters = (depth_img / 65535.0) * (depth_max - depth_min) + depth_min

    # depth_meters = depth_meters * 1000.0  # Convert to millimeters if needed

    # print("Max depth value: ", np.max(depth_meters))
    # print("Min depth value: ", np.min(depth_meters))

    rgb_img = cv2.imread(rgb_file_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    height, width = depth_img.shape[:2]
    
    # Convert FOV to radians and calculate focal length
    focal_length = 16 / np.tan(camera_fov / 2.0) # fov already in radians
    focal_length_px = focal_length * 512 / 32
    
    # Camera intrinsic parameters (assuming square pixels and centered principal point)
    cx = width / 2.0
    cy = height / 2.0
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Filter out invalid depth values (zeros or very large values)
    valid_mask = (depth_meters > 0) & (depth_meters < depth_max)  # Adjust max depth as needed

    if segmentation_mask is not None:
        seg_mask = cv2.imread(segmentation_mask, cv2.IMREAD_UNCHANGED)

        # ERODE THE MASK A BIT
        if erode_mask:
            kernel = np.ones((3, 3), np.uint8)
            seg_mask = cv2.erode(seg_mask, kernel, iterations=2)
        # debug save the mask
        # cv2.imwrite("debug_eroded_mask.png", seg_mask)

        seg_mask = (seg_mask > 127)
        seg_mask = seg_mask.astype(bool)
        valid_mask = valid_mask & seg_mask
    
    # Get valid pixels
    valid_u = u[valid_mask]
    valid_v = v[valid_mask]
    valid_depth = depth_meters[valid_mask]
    valid_rgb = rgb_img[valid_mask]
    
    # Reproject to 3D coordinates
    x = (valid_u - cx) * valid_depth / focal_length_px
    y = (valid_v - cy) * valid_depth / focal_length_px
    z = valid_depth
    
    # Stack coordinates
    points = np.column_stack((x, y, z))
    colors = valid_rgb.astype(np.uint8)

    print(f"Number of points after masking: {points.shape[0]}")

    # # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_camera_h = np.hstack([points, ones])

    # Convert your cam_transform to NumPy array if it isn't already
    c2w = np.array(cam_transform)  # shape (4, 4)
    c2w[:3, 1:3] *= -1
    # c2w = torch.from_numpy(c2w).float()
    # extrinsics = torch.inverse(c2w)
    # extrinsics = extrinsics.numpy()


    # # Apply the camera-to-world transformation
    points_world_h = (c2w @ points_camera_h.T).T # shape (N, 4)

    # # Drop the homogeneous coordinate to get (x, y, z)
    points_world = points_world_h[:, :3]

    return points_world, colors

def get_transform(json_path, rgb_file_name):
    with open(json_path, 'r') as f:
        data = json.load(f)
        frames = data['frames']
        for frame in frames:
            if frame['file_path'] == rgb_file_name:
                return frame['transform_matrix']
    return None
import argparse

parser = argparse.ArgumentParser(description="Reproject depth maps to 3D point clouds.")
parser.add_argument('--parent_dir', type=str, default='', help='Parent directory containing subdirectories with renderings.')
parser.add_argument('--output_dir', type=str, default='', help='Output directory to save point clouds.')
parser.add_argument('--segmentation_subject', type=str, default='object', choices=['person', 'object'], help='Subject type for segmentation mask: "person" or "object".')
args = parser.parse_args()
    
fov = 40/180 * np.pi
output_dir = args.output_dir
parent_dir = args.parent_dir
segmentation_subject = args.segmentation_subject

for render_dir_name in os.listdir(parent_dir):
    render_dir = os.path.join(parent_dir, render_dir_name)
    if not os.path.isdir(render_dir):
        continue

    print(f"Processing directory: {render_dir}")

    
    output_dir_full = os.path.join(output_dir, render_dir_name)
    os.makedirs(output_dir_full, exist_ok=True)

    transformers_json_path = f"{render_dir}/transforms.json"
    pc_list = []
    colors_list = []
    for file_path in os.listdir(render_dir):
        if not 'depth' in file_path and not 'mask' in file_path and file_path.endswith('.png'):
            rgb_file_path = os.path.join(render_dir, file_path)
            file_name = file_path.split('.')[0]
            depth_file_path = os.path.join(render_dir, file_name + '_depth.png')
            seg_mask = f"{render_dir}/mask_{segmentation_subject}_{file_name}.png"

            # check if segmentation mask exists
            if not os.path.exists(seg_mask):
                print(f"Mask {seg_mask} does not exist. Skipping.")
                continue

            cam_transform = get_transform(transformers_json_path, file_path)

            print(f"Processing {rgb_file_path} with depth {depth_file_path}")

            pc, colors = pc_from_depth(depth_file_path, rgb_file_path, fov, cam_transform, seg_mask, erode_mask=True)

            pc_list.append(pc)
            colors_list.append(colors)
    
    if len(pc_list) == 0:
        print(f"No point clouds were generated for directory {render_dir}. Skipping saving.")
        continue
    pc_all = np.concatenate(pc_list, axis=0)
    colors_all = np.concatenate(colors_list, axis=0)
    point_cloud = trimesh.PointCloud(vertices=pc_all, colors=colors_all)
    output_path = os.path.join(output_dir_full, f'{segmentation_subject}_pointcloud.ply')
    point_cloud.export(output_path)