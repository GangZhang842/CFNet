import torch
import torch.nn as nn
import torch.nn.functional as F


def reproj(pcds_xyzi, Voxel, dx, dy, phi_range_radian, theta_range_radian, dphi, dtheta):
    '''
    Input:
        pcds_xyzi (BS, N, 3), 3 -> (x, y, z)
    Output:
        pcds_coord_wl_reproj, pcds_sphere_coord_reproj (BS, N, 2, 1)
    '''
    # bev quat (BS, N)
    x = pcds_xyzi[:, :, 0]
    y = pcds_xyzi[:, :, 1]
    z = pcds_xyzi[:, :, 2]
    d = (x.pow(2) + y.pow(2) + z.pow(2)).sqrt() + 1e-12

    x_quan = (x - Voxel.range_x[0]) / dx
    y_quan = (y - Voxel.range_y[0]) / dy
    pcds_coord_wl_reproj = torch.stack((x_quan, y_quan), dim=-1).unsqueeze(-1)

    # rv quat
    phi = phi_range_radian[1] - torch.atan2(x, y)
    phi_quan = (phi / dphi)
    
    theta = theta_range_radian[1] - torch.asin(z / d)
    theta_quan = (theta / dtheta)
    pcds_sphere_coord_reproj = torch.stack((theta_quan, phi_quan), dim=-1).unsqueeze(-1)
    return pcds_coord_wl_reproj, pcds_sphere_coord_reproj


def reproj_with_offset(pcds_xyzi, pcds_offset, Voxel, dx, dy, phi_range_radian, theta_range_radian, dphi, dtheta):
    '''
    Input:
        pcds_xyzi (BS, 7, N, 1), 7 -> (x, y, z, intensity, dist, diff_x, diff_y)
        pcds_offset (BS, N, 3)
    Output:
        pcds_coord_wl_reproj, pcds_sphere_coord_reproj (BS, N, 2, 1)
    '''
    # bev quat (BS, N, 1)
    x = pcds_xyzi[:, 0, :] + pcds_offset[:, :, [0]]
    y = pcds_xyzi[:, 1, :] + pcds_offset[:, :, [1]]
    z = pcds_xyzi[:, 2, :] + pcds_offset[:, :, [2]]
    d = (x.pow(2) + y.pow(2) + z.pow(2)).sqrt() + 1e-12

    x_quan = (x - Voxel.range_x[0]) / dx
    y_quan = (y - Voxel.range_y[0]) / dy
    pcds_coord_wl_reproj = torch.stack((x_quan, y_quan), dim=2)

    # rv quat
    phi = phi_range_radian[1] - torch.atan2(x, y)
    phi_quan = (phi / dphi)
    
    theta = theta_range_radian[1] - torch.asin(z / d)
    theta_quan = (theta / dtheta)
    pcds_sphere_coord_reproj = torch.stack((theta_quan, phi_quan), dim=2)
    return pcds_coord_wl_reproj, pcds_sphere_coord_reproj