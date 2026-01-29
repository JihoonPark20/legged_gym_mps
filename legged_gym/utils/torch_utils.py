"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import torch
import numpy as np

# Check if MPS is available and disable JIT for MPS
_USE_JIT = not (torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)

def _jit_script_if_enabled(func):
    """Conditionally apply JIT script decorator based on device support"""
    if _USE_JIT:
        return torch.jit.script(func)
    return func

def to_torch(x, dtype=torch.float, device='mps', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def _quat_mul_impl(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

quat_mul = _jit_script_if_enabled(_quat_mul_impl)

def _normalize_impl(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

normalize = _jit_script_if_enabled(_normalize_impl)

def _quat_apply_impl(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

quat_apply = _jit_script_if_enabled(_quat_apply_impl)

def _quat_rotate_impl(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

quat_rotate = _jit_script_if_enabled(_quat_rotate_impl)

def _quat_rotate_inverse_impl(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

quat_rotate_inverse = _jit_script_if_enabled(_quat_rotate_inverse_impl)

def _quat_conjugate_impl(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

quat_conjugate = _jit_script_if_enabled(_quat_conjugate_impl)

def _quat_unit_impl(a):
    return normalize(a)

quat_unit = _jit_script_if_enabled(_quat_unit_impl)

def _quat_from_angle_axis_impl(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))

quat_from_angle_axis = _jit_script_if_enabled(_quat_from_angle_axis_impl)

def _normalize_angle_impl(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

normalize_angle = _jit_script_if_enabled(_normalize_angle_impl)

def _tf_inverse_impl(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)

tf_inverse = _jit_script_if_enabled(_tf_inverse_impl)

def _tf_apply_impl(q, t, v):
    return quat_apply(q, v) + t

tf_apply = _jit_script_if_enabled(_tf_apply_impl)

def _tf_vector_impl(q, v):
    return quat_apply(q, v)

tf_vector = _jit_script_if_enabled(_tf_vector_impl)

def _tf_combine_impl(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1

tf_combine = _jit_script_if_enabled(_tf_combine_impl)

def _get_basis_vector_impl(q, v):
    return quat_rotate(q, v)

get_basis_vector = _jit_script_if_enabled(_get_basis_vector_impl)


def get_axis_params(value, axis_idx, x_value=0., dtype=np.float64, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


def _copysign_impl(a, b):
    a_tensor = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a_tensor) * torch.sign(b)

copysign = _jit_script_if_enabled(_copysign_impl)

def _get_euler_xyz_impl(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

get_euler_xyz = _jit_script_if_enabled(_get_euler_xyz_impl)

def _quat_from_euler_xyz_impl(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)

quat_from_euler_xyz = _jit_script_if_enabled(_quat_from_euler_xyz_impl)

def _torch_rand_float_impl(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower

torch_rand_float = _jit_script_if_enabled(_torch_rand_float_impl)

def _torch_random_dir_2_impl(shape, device):
    angle = torch_rand_float(-np.pi, np.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)

torch_random_dir_2 = _jit_script_if_enabled(_torch_random_dir_2_impl)

def _tensor_clamp_impl(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)

tensor_clamp = _jit_script_if_enabled(_tensor_clamp_impl)

def _scale_impl(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)

scale = _jit_script_if_enabled(_scale_impl)

def _unscale_impl(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

unscale = _jit_script_if_enabled(_unscale_impl)


def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
