"""
Author: 李子璇
目的：筛选需要点
"""

import numpy as np
import os
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyElement, PlyData


def save_to_ply(points, output_path, normals=None, colors=None):
    """
    将点云数据保存为PLY文件

    参数:
    points: numpy数组，形状(N, 3)，点云的xyz坐标
    output_path: 字符串，输出文件路径
    """

    print(f"正在保存 {points.shape[0]} 个匹配点到 {output_path}")
    if normals is None:
        normals = np.zeros((points.shape[0], 3), dtype=np.float32)
    # 创建结构化数组，包含xyz坐标
    vertex_data = np.array([tuple(list(point) + list(normals) + list(color))
                           for point, normals, color in zip(points, normals, colors)],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    # 创建PLY元素和数据对象
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    ply_data = PlyData([vertex_element])

    # 写入文件
    ply_data.write(output_path)
    print(f"PLY文件保存成功: {output_path}")


def match_point_clouds(reference_cloud, target_cloud, output_path):
    """
    将参考点云中的每个点与目标点云中最近的点进行匹配

    参数:
    reference_cloud: numpy数组，形状(N, 3)，参考点云的xyz坐标
    target_cloud: numpy数组，形状(M, 3)，目标点云的xyz坐标
    output_path: 字符串，输出PLY文件的路径

    返回:
    matched_points: numpy数组，匹配后的点云
    distances: numpy数组，每个匹配点对的距离
    """
    if isinstance(target_cloud, o3d.geometry.PointCloud):
        target_points = np.asarray(target_cloud.points)
        target_normals = np.asarray(target_cloud.normals) if target_cloud.has_normals() else None
        target_colors = np.asarray(target_cloud.colors)
    else:
        target_points = target_cloud
        target_normals = None
        target_colors = None

    if isinstance(reference_cloud, o3d.geometry.PointCloud):
        reference_points = np.asarray(reference_cloud.points)
    else:
        reference_points = reference_cloud
    result = {
        'output_file': None,
        'matched_points': None,
        'distances': None,
    }

    # 使用KD树进行高效的最近邻搜索
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_points)
    distances, indices = nbrs.kneighbors(reference_points)

    # 获取匹配的点
    matched_points = target_points[indices.flatten()]
    matched_normals = target_normals[indices.flatten()] if target_normals is not None else None
    matched_colors = target_colors[indices.flatten()] if target_colors is not None else None
    distances = distances.flatten()

    # 保存为PLY文件
    output_file_path = os.path.join(output_path, "points3D.ply")
    os.makedirs(output_path, exist_ok=True)
    save_to_ply(matched_points, output_file_path, matched_normals, matched_colors)

    result['output_file'] = output_file_path
    result['matched_points'] = matched_points
    result['distances'] = distances

    return result






