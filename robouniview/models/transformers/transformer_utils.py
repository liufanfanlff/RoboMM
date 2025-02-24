import copy
import math
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import cv2
from math import sin, cos


def parse_numpy_dtype(dtype: str):
    if dtype == "np.long":
        return  np.longlong
    elif dtype == "np.int":
        return  np.int_
    elif dtype == "np.float":
        return  np.float_
    elif dtype == "np.bool":
        return np.bool_
    else:
        raise NotImplementedError(f"Got unexpected dtype in numpy parser: {dtype}")

def euler_to_Rot(yaw, pitch, roll):
    P = np.array([[cos(pitch), 0, sin(pitch)],
                  [0, 1, 0],
                  [-sin(pitch), 0, cos(pitch)]])
    R = np.array([[1, 0, 0],
                  [0, cos(roll), -sin(roll)],
                  [0, sin(roll), cos(roll)]])
    Y = np.array([[cos(yaw), -sin(yaw), 0],
                  [sin(yaw), cos(yaw), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def get_undistorted_point(point, cameraMatrix, distCoeffs):
    pts = np.array(point).reshape((-1, 1, 2)).astype(parse_numpy_dtype("np.float"))
    undistorted_pts = cv2.undistortPoints(
        pts, cameraMatrix, distCoeffs, P=cameraMatrix)
    return np.squeeze(undistorted_pts)


def unproject_2d_to_3d(pt_2d, depth, P):
    z = depth - P[2, 3]
    x = (pt_2d[:, 0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
    y = (pt_2d[:, 1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
    pt_3d = np.array([x, y, z], dtype=np.float32).transpose((1, 0))
    return pt_3d


def encode_grid_to_emb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


def encode_grid_to_emb2d(x_shape, device, temperature=10000, normalize=True):
    scale = 2 * math.pi
    b, c, h, w = x_shape
    num_pos_feats = c // 2
    ones = torch.ones(size=(b, h, w), device=device)
    y_embed = ones.cumsum(1, dtype=torch.float32)
    x_embed = ones.cumsum(2, dtype=torch.float32)

    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # length=num_pos_feats

    pos_x = x_embed[:, :, :, None] / dim_t  # b,h,w,num_pos_feats
    pos_y = y_embed[:, :, :, None] / dim_t  # b,h,w,num_pos_feats

    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    return pos


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def get_distorted_point(point, cameraMatrix, distCoeffs):
    pts = np.array(point).reshape((-1, 1, 2)).astype(parse_numpy_dtype("np.float"))
    rtemp = ttemp = np.array([0, 0, 0], dtype='float32')
    # note that we have to normalize the undirstorted_pts wrt camera matrix
    pts[:, :, 0] = (pts[:, :, 0] - cameraMatrix[0, 2]) / cameraMatrix[0, 0]
    pts[:, :, 1] = (pts[:, :, 1] - cameraMatrix[1, 2]) / cameraMatrix[1, 1]
    pts_3d = cv2.convertPointsToHomogeneous(pts)
    distorted_pts, _ = cv2.projectPoints(pts_3d, rtemp, ttemp, cameraMatrix, distCoeffs)
    return np.squeeze(distorted_pts)


def create_uv_grid(uv_range, resolution, use_frustum_refpoints):
    uv_shape = [np.round((s - e) / r).astype(parse_numpy_dtype("np.int")) for s, e, r in zip(uv_range[::2], uv_range[1::2], resolution)]
    x = np.linspace(uv_range[0] - resolution[0] / 2, uv_range[1] + resolution[0] / 2, abs(uv_shape[0]))
    y = np.linspace(uv_range[2] - resolution[1] / 2, uv_range[3] + resolution[1] / 2, abs(uv_shape[1]))
    z = np.linspace(uv_range[4] - resolution[2] / 2, uv_range[5] + resolution[2] / 2, abs(uv_shape[2]))
    uv_grid_y, uv_grid_x = np.meshgrid(y, x)  # rigY H x rigX W
    uv_grid = []
    for h in z:
        z_grid = np.ones_like(uv_grid_x) * h
        uv_grid_2d = np.stack((uv_grid_x, uv_grid_y, z_grid), axis=2)
        uv_grid.append(uv_grid_2d)
    uv_grid = np.array(uv_grid)
    uv_grid = uv_grid.transpose(1, 2, 0, 3)
    if use_frustum_refpoints:
        max_dis = (uv_range[0] ** 2 + uv_range[2] ** 2) ** .5
        dis = (uv_grid[..., 0] ** 2 + uv_grid[..., 1] ** 2) ** .5
        uv_grid[..., 2] = uv_grid[..., 2] * (1 + dis / max_dis)
    scalar_grid = np.expand_dims(np.ones_like(uv_grid[..., 0]), 3)
    uv_grid = np.concatenate((uv_grid, scalar_grid), axis=3)
    return uv_grid


def project_grid_image(uv_grid, intrinsic, extrinsic, distortion, height, width):
    uv_shape = uv_grid.shape[:3]
    pt3d = uv_grid.transpose((3, 0, 1, 2)).reshape(4, -1) # 3D 网格点[-0.525,-0.525,-0.1]-[0.525,0.525,1.1]
    undist_cam_pt = np.dot(intrinsic, np.dot(extrinsic, pt3d)[:3]) # 映射到图片上
    depth_valid = undist_cam_pt[2, :] > 0
    scale = undist_cam_pt[2]
    scale_valid = scale > 0
    valid = depth_valid * scale_valid

    img_pts = np.arange(height)
    img_pts = np.vstack((np.zeros_like(img_pts), img_pts)).T # 图片左侧UV位置
    undistorted_img_pts = get_undistorted_point(img_pts, intrinsic, distortion) # 去畸变
    undistorted_img_pts_min = undistorted_img_pts[:, 0].min()

    img_pts = np.arange(height)
    img_pts = np.vstack((np.ones_like(img_pts) * width, img_pts)).T
    undistorted_img_pts = get_undistorted_point(img_pts, intrinsic, distortion)
    undistorted_img_pts_max = undistorted_img_pts[:, 0].max()

    undist_cam_pt = np.array([undist_cam_pt[0] / scale, undist_cam_pt[1] / scale]) # /z,
    undist_cam_pt = undist_cam_pt.transpose()
    dist_cam_pt = get_distorted_point(undist_cam_pt, intrinsic, distortion)

    undistorted_cam_pt_valid = (undistorted_img_pts_min < undist_cam_pt[:, 0]) * \
                               (undist_cam_pt[:, 0] < undistorted_img_pts_max)
    x_valid = (0 < dist_cam_pt[:, 0]) * (dist_cam_pt[:, 0] < width)
    y_valid = (0 < dist_cam_pt[:, 1]) * (dist_cam_pt[:, 1] < height)
    valid = x_valid * y_valid * valid * undistorted_cam_pt_valid
    valid = valid.reshape(uv_shape[0], uv_shape[1], uv_shape[2]).astype(int)
    dist_cam_pt = dist_cam_pt.reshape(uv_shape[0], uv_shape[1], uv_shape[2], -1).astype(int)
    return dist_cam_pt, valid

    if 0:
        import open3d as o3d
        from robouniview.data.data_utils import OccupancyVFE
        pcd = np.load('tmp.npy')
        voxel_range = [[-2.0, 2.0], [-1.0, 3.0], [0.0, 1.0]]
        voxel_size = [0.0125*4, 0.0125*4, 0.0125*2]
        vfe_generator = OccupancyVFE(voxel_range, voxel_size) # 重新定义区域大小，导航需要更大的范围
        point_tmp, rgb_tmp = vfe_generator.decode_occupied_grid_with_range(pcd)
        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(point_tmp)
        _pcd.colors = o3d.utility.Vector3dVector(rgb_tmp)
        o3d.io.write_point_cloud("tmp.pcd", _pcd)

        im = np.zeros([int(height), int(width), 3], dtype=np.uint8)
        for uv, isv, rbg in zip(dist_cam_pt, valid, rgb_tmp):
            if isv:
                im[int(uv[0]), int(uv[1]), :] = ((rbg *(0.26862954, 0.26130258, 0.27577711))+(0.48145466, 0.4578275, 0.40821073))*255


def project_sparse_grid_image(sparse_grid, intrinsic, extrinsic, distortion, height, width):
    pt3d = sparse_grid.transpose((1, 0))  # N,4 --> 4,N
    undist_cam_pt = np.dot(intrinsic, np.dot(extrinsic, pt3d)[:3])
    depth_valid = undist_cam_pt[2, :] > 0
    scale = undist_cam_pt[2]
    scale_valid = scale > 0
    valid = depth_valid * scale_valid

    undist_cam_pt = np.array([undist_cam_pt[0] / scale, undist_cam_pt[1] / scale])
    undist_cam_pt = undist_cam_pt.transpose()
    dist_cam_pt = get_distorted_point(undist_cam_pt, intrinsic, distortion)

    x_valid = (0 < dist_cam_pt[:, 0]) * (dist_cam_pt[:, 0] < width)
    y_valid = (0 < dist_cam_pt[:, 1]) * (dist_cam_pt[:, 1] < height)
    valid = x_valid * y_valid * valid
    valid = valid.astype(int)
    return dist_cam_pt, valid


def unpack_calib(calib_batch: dict, bs):
    """
    Convert a dict with values of size b*3*3 to
    a list of b dicts with values of size 3*3.
    """
    calib_list = []

    #calib_bs = calib_batch['rgb_static']['intrinsic_matrix'].shape[0]
    for b in range(bs):
        curr_calib = dict()
        # for video pipeline. batch size is not the same as single frame.
        # if bs > calib_bs:
        #     b = b // (bs // calib_bs)
        for k, v in calib_batch.items():
            if k == 'vehicle_name':
                curr_calib[k] = v[b]
            else:  # cam
                curr_calib[k] = dict()
                for mat_k, mat_v in v.items():  # intrinsic/extrinsic/distortion
                    curr_calib[k][mat_k] = mat_v[b]  # 0 is for single frame
        calib_list.append(curr_calib)

    return calib_list
