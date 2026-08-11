"""
panorama_utils.py
Equirectangular ↔ Perspective conversions using only numpy + cv2.
No external dependencies.
"""

import numpy as np
import cv2


def equirect_to_perspective(equi_img, fov_deg=90.0, out_h=512, out_w=512,
                            yaw=0.0, pitch=0.0, roll=0.0):
    """
    Extract a perspective sub-view from an equirectangular panorama.
    
    Args:
        equi_img: np.ndarray (H, W, C) uint8/float — BGR/RGB/depth
        fov_deg: horizontal field of view in degrees
        out_h, out_w: output perspective image size
        yaw, pitch, roll: camera orientation in radians
        
    Returns:
        persp_img: np.ndarray (out_h, out_w, C) or (out_h, out_w)
    """
    equi_h, equi_w = equi_img.shape[:2]
    fov = np.deg2rad(fov_deg)
    
    # Intrinsic parameters for perspective camera
    fx = (out_w / 2.0) / np.tan(fov / 2.0)
    fy = fx
    cx = out_w / 2.0
    cy = out_h / 2.0
    
    # Grid of pixel coordinates in perspective image
    x = (np.arange(out_w) - cx) / fx
    y = (np.arange(out_h) - cy) / fy
    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx)
    
    # Rays in camera space (z forward)
    rays_cam = np.stack([xx, yy, zz], axis=-1)
    rays_cam = rays_cam / np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    
    # Rotation matrix
    cy_r, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    
    R = np.array([
        [cy_r*cp, cy_r*sp*sr - sy*cr, cy_r*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy_r*cr, sy*sp*cr - cy_r*sr],
        [-sp,   cp*sr,              cp*cr             ]
    ], dtype=np.float64)
    
    rays_world = rays_cam @ R.T
    
    # Convert to spherical coordinates
    xw, yw, zw = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]
    azimuth = np.arctan2(xw, zw)
    elevation = np.arctan2(yw, np.sqrt(xw**2 + zw**2))
    
    # Map to equirectangular pixel coordinates
    map_x = (azimuth / np.pi + 1.0) / 2.0 * (equi_w - 1)
    map_y = (1.0 - (elevation / (np.pi/2) + 1.0) / 2.0) * (equi_h - 1)
    
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    
    is_single = (equi_img.ndim == 2)
    if is_single:
        equi_img = equi_img[..., np.newaxis]
    
    persp = cv2.remap(equi_img, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT,
                      borderValue=0)
    
    if is_single:
        persp = persp[..., 0]
    
    return persp


def perspective_to_equirect(persp_depths, params, equi_h, equi_w):
    """
    Stitch multiple perspective depth maps back into an equirectangular map.
    
    Args:
        persp_depths: list of np.ndarray (out_h, out_w) — predicted depth per sub-view
        params: list of dicts [{'yaw':..., 'pitch':..., 'roll':..., 'fov_deg':...,
                                'out_h':..., 'out_w':...}, ...]
        equi_h, equi_w: output equirectangular size
        
    Returns:
        equi_depth: np.ndarray (equi_h, equi_w) float32
    """
    accum = np.zeros((equi_h, equi_w), dtype=np.float64)
    count = np.zeros((equi_h, equi_w), dtype=np.float64)
    
    for depth_persp, p in zip(persp_depths, params):
        fov = np.deg2rad(p['fov_deg'])
        out_h, out_w = p['out_h'], p['out_w']
        fx = (out_w / 2.0) / np.tan(fov / 2.0)
        fy = fx
        cx = out_w / 2.0
        cy = out_h / 2.0
        
        # Grid over equirectangular image
        u = np.arange(equi_w)
        v = np.arange(equi_h)
        uu, vv = np.meshgrid(u, v)
        
        # Spherical coords
        azimuth = (uu / (equi_w - 1)) * 2.0 * np.pi - np.pi
        elevation = (1.0 - vv / (equi_h - 1)) * np.pi - np.pi/2
        
        # 3D direction in world space
        xw = np.cos(elevation) * np.sin(azimuth)
        yw = np.sin(elevation)
        zw = np.cos(elevation) * np.cos(azimuth)
        rays_world = np.stack([xw, yw, zw], axis=-1)
        
        # Rotate to camera space (inverse rotation)
        yaw, pitch, roll = p['yaw'], p['pitch'], p['roll']
        cy_r, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        
        R = np.array([
            [cy_r*cp, cy_r*sp*sr - sy*cr, cy_r*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy_r*cr, sy*sp*cr - cy_r*sr],
            [-sp,   cp*sr,              cp*cr             ]
        ], dtype=np.float64)
        
        rays_cam = rays_world @ R
        
        # Project to perspective image plane
        xc, yc, zc = rays_cam[..., 0], rays_cam[..., 1], rays_cam[..., 2]
        
        valid = zc > 1e-6
        u_persp = np.zeros_like(xc)
        v_persp = np.zeros_like(yc)
        
        u_persp[valid] = xc[valid] / zc[valid] * fx + cx
        v_persp[valid] = yc[valid] / zc[valid] * fy + cy
        
        # Check bounds
        in_bounds = valid & (u_persp >= 0) & (u_persp < out_w - 1) & \
                    (v_persp >= 0) & (v_persp < out_h - 1)
        
        map_x = u_persp.astype(np.float32)
        map_y = v_persp.astype(np.float32)
        
        sampled = cv2.remap(depth_persp, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
        
        accum += sampled
        count += (sampled > 0).astype(np.float64)
    
    equi_depth = np.zeros((equi_h, equi_w), dtype=np.float32)
    np.divide(accum, count, out=equi_depth, where=count > 0)
    return equi_depth


def get_perspective_params(n_azimuth=8, fov_deg=90.0, out_h=512, out_w=512,
                           pitch=0.0, roll=0.0):
    """
    Generate parameters for N perspective sub-views evenly spaced around azimuth.
    """
    params = []
    for i in range(n_azimuth):
        yaw = (i / n_azimuth) * 2.0 * np.pi - np.pi
        params.append({
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'fov_deg': fov_deg,
            'out_h': out_h,
            'out_w': out_w,
        })
    return params


def infer_depth_panoramic(da2_model, equi_bgr, n_views=8, fov_deg=90.0,
                          persp_h=512, persp_w=512, device=None):
    """
    Run Depth Anything V2 on an equirectangular panorama.
    """
    equi_h, equi_w = equi_bgr.shape[:2]
    
    # 1. Extract perspective views
    params = get_perspective_params(n_azimuth=n_views, fov_deg=fov_deg,
                                    out_h=persp_h, out_w=persp_w)
    persp_depths = []
    
    for p in params:
        persp = equirect_to_perspective(
            equi_bgr,
            fov_deg=p['fov_deg'],
            out_h=p['out_h'],
            out_w=p['out_w'],
            yaw=p['yaw'],
            pitch=p['pitch'],
            roll=p['roll']
        )
        # persp is (H, W, 3) BGR
        depth = da2_model.predict(persp)
        persp_depths.append(depth)
    
    # 2. Stitch back to equirectangular
    equi_depth = perspective_to_equirect(persp_depths, params,
                                         equi_h=equi_h, equi_w=equi_w)
    return equi_depth