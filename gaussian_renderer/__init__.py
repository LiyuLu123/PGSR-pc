#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.app_model import AppModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           app_model: AppModel=None, return_plane = True, return_depth_normal = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features # shs.shape ([52093, 16, 3])
    else:
        colors_precomp = override_color
        
    # # 在render函数中shs被赋值后添加：
    # if shs is not None:
    #     print("First 3 Gaussians SH coefficients:")
    #     print(shs[:3])  # 显示前3个高斯点的所有球谐系数
    #
    #     # 或者只看第一个高斯点的详细信息
    #     print("First Gaussian detailed SH:")
    #     print(shs[0])
    # exit()
    # if colors_precomp is None:
    #     # 调试：打印第一个高斯点的完整信息
    #     debug_first_gaussian = True
    #     if debug_first_gaussian and pc.get_features.shape[0] > 0:
    #         print("\n" + "=" * 60)
    #         print("DEBUG: First Gaussian Point Analysis")
    #         print("=" * 60)
    #
    #         # 1. 基本信息
    #         print(f"Total gaussians: {pc.get_features.shape[0]}")
    #         print(f"Feature dimension: {pc.get_features.shape[1]}")
    #         print(f"Max SH degree: {pc.max_sh_degree}")
    #         print(f"Active SH degree: {pc.active_sh_degree}")
    #
    #         # 2. 原始球谐系数
    #         first_features = pc.get_features[0]
    #         print(f"\nRaw SH coefficients shape: {first_features.shape}")
    #         print(f"Raw values: {first_features}")
    #
    #         # 3. 按通道分解（如果可能）
    #         coeff_per_channel = (pc.max_sh_degree + 1) ** 2
    #         expected_total = 3 * coeff_per_channel
    #         print(f"Expected coefficients per channel: {coeff_per_channel}")
    #         print(f"Expected total coefficients: {expected_total}")
    #
    #         if len(first_features) == expected_total:
    #             for i, channel in enumerate(['R', 'G', 'B']):
    #                 start = i * coeff_per_channel
    #                 end = (i + 1) * coeff_per_channel
    #                 coeffs = first_features[start:end]
    #                 print(f"{channel} channel coefficients: {coeffs}")
    #         else:
    #             print(f"Warning: Actual feature size ({len(first_features)}) != expected ({expected_total})")
    #
    #         print("=" * 60 + "\n")

    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    #
    #         # 打印重排后的第一个高斯点球谐系数
    #         print("=== First Gaussian SH Coefficients (Reshaped) ===")
    #         print(f"shs_view shape: {shs_view.shape}")
    #         print(f"First gaussian shs_view:")
    #         print(shs_view[0])  # 形状为 (3, (max_sh_degree+1)²)
    #         print("R channel:", shs_view[0, 0, :])
    #         print("G channel:", shs_view[0, 1, :])
    #         print("B channel:", shs_view[0, 2, :])
    #         print("=" * 50)
    # exit()

    return_dict = None
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    if not return_plane:
        rendered_image, radii, out_observe, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe}
        if app_model is not None and pc.use_app:
            appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
            app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
            return_dict.update({"app_image": app_image})
        return return_dict

    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    depth_z = pts_in_cam[:, 2]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance

    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp)

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]
    
    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "rendered_normal": rendered_normal,
                    "plane_depth": plane_depth,
                    "rendered_distance": rendered_distance
                    }
    
    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
        app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
        return_dict.update({"app_image": app_image})   

    if return_depth_normal:
        depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
        return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict