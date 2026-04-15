import open3d as o3d
import numpy as np
import argparse
import os

def clean_pointcloud_A(cloud_A, cloud_B, threshold=0.01):
    # Convert to numpy arrays
    points_A = np.asarray(cloud_A.points)
    points_B = np.asarray(cloud_B.points)

    # Create KDTree for fast nearest-neighbor search
    pcd_tree_B = o3d.geometry.KDTreeFlann(cloud_B)

    # Keep indices of points from A that are NOT within threshold distance to any point in B
    keep_indices = []
    for i, p in enumerate(points_A):
        [k, idx, dist] = pcd_tree_B.search_knn_vector_3d(p, 1)
        if dist[0] > threshold**2:  # distance squared comparison
            keep_indices.append(i)

    # Use select_by_index to retain all attributes (colors, normals, etc.)
    filtered_cloud_A = cloud_A.select_by_index(keep_indices)

    return filtered_cloud_A

parser = argparse.ArgumentParser(description="Remove intersections between two point clouds.")
parser.add_argument('--base_dir', type=str, default='', help='Base directory containing point clouds.')
parser.add_argument('--threshold', type=float, default=0.01, help='Distance threshold to consider points as intersecting.')
args = parser.parse_args()

for subdir in os.listdir(args.base_dir):
    subdir_full = os.path.join(args.base_dir, subdir)
    if not os.path.isdir(subdir_full):
        continue
    print(f"Processing directory: {subdir_full}")
    # Load point clouds
    object_pc_path = os.path.join(subdir_full, "object_pointcloud.ply")
    person_pc_path = os.path.join(subdir_full, "person_pointcloud.ply")
    if not os.path.exists(object_pc_path) or not os.path.exists(person_pc_path):
        print(f"Point cloud files not found in {subdir_full}. Skipping.")
        continue
    object_pc = o3d.io.read_point_cloud(object_pc_path)
    person_pc = o3d.io.read_point_cloud(person_pc_path)


    clean_person = clean_pointcloud_A(person_pc, object_pc, threshold=args.threshold)
    clean_object = clean_pointcloud_A(object_pc, person_pc, threshold=args.threshold)

    o3d.io.write_point_cloud(os.path.join(subdir_full, "object_pointcloud_clean.ply"), clean_object)
    o3d.io.write_point_cloud(os.path.join(subdir_full, "person_pointcloud_clean.ply"), clean_person)
