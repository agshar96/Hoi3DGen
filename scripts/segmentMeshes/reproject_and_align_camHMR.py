import numpy as np
import cv2
import trimesh
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch3d.loss import chamfer_distance
import os
import argparse 
from pathlib import Path

def scale_source_mesh(source_mesh, target_pc_path, transform=None):
    # get source scale
    source_origin_transform, _ = trimesh.bounds.oriented_bounds(source_mesh)


    # get target scale
    target_mesh = trimesh.load(target_pc_path)

    target_origin_transform, _ = trimesh.bounds.oriented_bounds(target_mesh)

    source_mesh.apply_transform(source_origin_transform) # this brings the mesh to axis aligned bounding box
    target_mesh.apply_transform(target_origin_transform)
    if transform is not None:
        transform = source_origin_transform @ transform


    # Recalculate extents after aligning to bounding box
    source_extents = source_mesh.bounding_box.extents
    target_extents = target_mesh.bounding_box.extents

    source_sorted_indices = np.argsort(source_extents)
    target_sorted_indices = np.argsort(target_extents)


    # compute per axis scale factors
    scale_factors = []

    for i in range(3):
        source_extent = source_extents[i]
        source_order_index = source_sorted_indices[i]

        target_order_index = np.where(target_sorted_indices == source_order_index)[0][0]
        # print(f"Source axis {i} (extent {source_extent}) corresponds to target axis {target_order_index}")
        target_extent = target_extents[target_order_index]

        scale_factor = target_extent / source_extent
        scale_factors.append(scale_factor)
        # print(f"Axis {i}: Source extent = {source_extent}, Target extent = {target_extent}, Scale factor = {scale_factor}")

    scale_factors = np.array(scale_factors).reshape(1, 3)

    # Build scaling matrix
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] = np.diag(scale_factors.flatten())

    # apply scaling
    source_mesh.apply_transform(scale_matrix)
    source_mesh.apply_transform(np.linalg.inv(source_origin_transform)) # bring back to original orientation

    if transform is not None:
        transform = np.linalg.inv(source_origin_transform) @ scale_matrix @ transform

    # save the scaled object as obj plus mtl file for textures
    # os.makedirs("temp_test", exist_ok=True)
    # source_mesh.export("temp_test/mesh_scaled.obj")

    return source_mesh, transform

def rodrigues_rotation_matrix(r):
    """Convert axis-angle vector to rotation matrix using Rodrigues' formula."""
    theta = torch.norm(r, dim=-1, keepdim=True) + 1e-8
    k = r / theta
    K = torch.zeros(r.shape[0], 3, 3, device=r.device)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] =  k[:, 1]
    K[:, 1, 0] =  k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] =  k[:, 0]
    I = torch.eye(3, device=r.device).unsqueeze(0).expand(r.shape[0], -1, -1)
    R = I + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta)).unsqueeze(-1) * (K @ K)
    return R

def translate_source_mesh(source_mesh, target_pc_path, transform=None):
    source_centroid = source_mesh.centroid
    target_mesh = trimesh.load(target_pc_path)
    target_centroid = target_mesh.centroid

    translation_vector = target_centroid - source_centroid

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation_vector

    source_mesh.apply_transform(translation_matrix)

    if transform is not None:
        transform = translation_matrix @ transform

    return source_mesh, transform

def downsample_points(points, n=5000):
    if points.shape[0] > n:
        idx = np.random.choice(points.shape[0], n, replace=False)
        return points[idx]
    return points

def align_source_mesh(source_mesh, target_pc_path):
    target_pc = trimesh.load(target_pc_path)
    target_pc_points = target_pc.vertices

    target_pc_points = downsample_points(target_pc_points, n=5000)

    source_vertices = source_mesh.vertices

    source_points = torch.tensor(source_vertices, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
    target_points = torch.tensor(target_pc_points, dtype=torch.float32).unsqueeze(0)  # (1, M, 3)

    translation = nn.Parameter(torch.zeros(1, 1, 3))  # shape (1,1,3)
    scale = nn.Parameter(torch.ones(1, 1, 1))  # shape (1,1) # we want it to be a single number
    # Optimizer
    optimizer = optim.Adam([translation, scale], lr=1e-2)

    num_epochs = 50

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        transformed_source = scale * source_points + translation  # (1, N, 3)

        loss, _ = chamfer_distance(target_points, transformed_source)

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Translation: {translation.data.numpy()}, Scale: {scale.data.numpy()}")

    aligned_vertices = scale.detach().numpy().squeeze() * source_mesh.vertices + translation.detach().numpy().squeeze()
    source_mesh.vertices = aligned_vertices
    return source_mesh

def align_source_mesh_w_rotation(source_mesh, target_pc_path, transform=None):
    target_pc = trimesh.load(target_pc_path)
    target_pc_points = target_pc.vertices

    target_pc_points = downsample_points(target_pc_points, n=5000)

    source_vertices = source_mesh.vertices

    source_points = torch.tensor(source_vertices, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
    target_points = torch.tensor(target_pc_points, dtype=torch.float32).unsqueeze(0)  # (1, M, 3)
    

    translation = nn.Parameter(torch.zeros(1, 1, 3))  # shape (1,1,3)
    scale = nn.Parameter(torch.ones(1, 1, 1))  # shape (1,1) # we want it to be a single number
    rotation_vec = nn.Parameter(torch.zeros(1, 3))  # (1, 3)

    optimizer = optim.Adam([translation, rotation_vec, scale], lr=1e-2)

    num_epochs = 50

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        R = rodrigues_rotation_matrix(rotation_vec)  # (1, 3, 3)
        rotated_source = torch.matmul(source_points, R.transpose(1, 2))  # (1, N, 3)
        transformed_source = scale * rotated_source + translation  # (1, N, 3)

        loss, _ = chamfer_distance(target_points, transformed_source)

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Translation: {translation.data.numpy()}, Scale: {scale.data.numpy()}")
    
    aligned_vertices = scale.detach().numpy().squeeze() * (rodrigues_rotation_matrix(rotation_vec.detach()).squeeze().cpu().numpy() @ source_mesh.vertices.T).T + translation.detach().numpy().squeeze()

    if transform is not None:
        R = rodrigues_rotation_matrix(rotation_vec.detach()).squeeze().numpy()  # 3x3 rotation matrix
        s = scale.detach().numpy().squeeze()  # scalar
        t = translation.detach().numpy().squeeze()
        T = np.eye(4)
        T[:3, :3] = s * R
        T[:3, 3] = t
        transform = T @ transform

    source_mesh.vertices = aligned_vertices
    return source_mesh, transform

def get_transform(json_path, rgb_file_name):
    with open(json_path, 'r') as f:
        data = json.load(f)
        frames = data['frames']
        for frame in frames:
            if frame['file_path'] == rgb_file_name:
                return frame['transform_matrix']
    return None

def get_world_SMPL_mesh(camHMR_mesh_path, json_path, transform=None):

    frame_id = os.path.basename(camHMR_mesh_path).split('.')[0].split('_')[-1]
    camHMR_mesh = trimesh.load(camHMR_mesh_path)
    camHMR_vertices = camHMR_mesh.vertices
    camHMR_faces = camHMR_mesh.faces

    rgb_file_name = f"{frame_id}.png"

    cam_transform = get_transform(json_path, rgb_file_name)

    c2w = np.array(cam_transform)  # shape (4, 4)
    # c2w[:3, 1:3] *= -1

    vertices_h = np.hstack([camHMR_vertices, np.ones((camHMR_vertices.shape[0], 1))])  # shape (N, 4)
    vertices_world_h = (c2w @ vertices_h.T).T  # shape (N, 4)
    vertices_world = vertices_world_h[:, :3]  # shape (N, 3)

    if transform is not None:
        transform = c2w @ transform

    output_mesh = trimesh.Trimesh(vertices=vertices_world, faces=camHMR_faces)

    return output_mesh, transform


parser = argparse.ArgumentParser()
parser.add_argument('--input_mesh_dir', type=str, default='', help='Directory containing camHMR meshes')
parser.add_argument('--output_dir', type=str, default='', help='Directory to save world SMPL meshes')
parser.add_argument('--json_base_path', type=str, default='RENDERED_combined_3D', help='Base path for transforms.json files')
parser.add_argument('--target_pc_base_path', type=str, default='SEGMENTATION_OUTPUTS', help='Base path for target point clouds')
parser.add_argument('--use_rotation', type=bool, default=True, help='Whether to use rotation in alignment')
parser.add_argument('--skip_existing', type=bool, default=True, help='Whether to skip processing if output already exists')
parser.add_argument('--save_transforms', type=bool, default=True, help='Whether to save the computed transforms')
args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

fov = 40/180 * np.pi
run_dir = Path(args.input_mesh_dir).parent

for mesh_file in os.listdir(args.input_mesh_dir):
    if mesh_file.endswith('.obj'):
        instance_dir = f"{mesh_file.split('.')[0].split('_')[0]}_{mesh_file.split('.')[0].split('_')[1]}"
        print(f"Processing {instance_dir}")

        if args.save_transforms:
            # transform_file_path = os.path.join(args.input_mesh_dir, f"{mesh_file.split('.')[0]}_T.npz")
            # # read the transform file
            # if os.path.exists(transform_file_path):
            #     transform = np.load(transform_file_path)['rotation'] # it is a 4x4 matrix
            transform = np.eye(4)

        if args.skip_existing and os.path.exists(os.path.join(args.output_dir, instance_dir, "aligned_SMPL.obj")):
            print(f"Output for {instance_dir} already exists. Skipping.")
            continue

        json_path = os.path.join(run_dir, args.json_base_path, instance_dir, 'transforms.json')
        if not os.path.exists(json_path):
            print(f"Transforms file {json_path} does not exist. Skipping.")
            continue
        
        input_mesh_path = os.path.join(args.input_mesh_dir, mesh_file)
        if not os.path.exists(input_mesh_path):
            print(f"Mesh file {input_mesh_path} does not exist. Skipping.")
            continue
        # output_mesh_path = os.path.join(args.output_dir, instance_dir)

        output_mesh, transform = get_world_SMPL_mesh(input_mesh_path, json_path, transform if args.save_transforms else None) # if no save_transforms then it returns None for transform

        target_pc_path = os.path.join(run_dir, args.target_pc_base_path, instance_dir, "person_pointcloud_clean.ply")
        if not os.path.exists(target_pc_path):
            print(f"Target point cloud {target_pc_path} does not exist. Skipping.")
            continue

        output_mesh, transform = scale_source_mesh(output_mesh, target_pc_path, transform if args.save_transforms else None)

        output_mesh, transform = translate_source_mesh(output_mesh, target_pc_path, transform if args.save_transforms else None)

        if args.use_rotation:
            output_mesh, transform = align_source_mesh_w_rotation(output_mesh, target_pc_path, transform if args.save_transforms else None)
        else:
            output_mesh = align_source_mesh(output_mesh, target_pc_path)

        output_mesh_path = os.path.join(args.output_dir, instance_dir)
        os.makedirs(output_mesh_path, exist_ok=True)

        if args.save_transforms:
            np.savez_compressed(os.path.join(output_mesh_path, "SMPL_transform.npz"), T=transform)

        output_mesh.export(os.path.join(output_mesh_path, f"aligned_SMPL.obj"))
