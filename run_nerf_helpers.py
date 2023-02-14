import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def positional_encoding(input, L):
    period = (2. ** torch.linspace(0, L-1, L))
    embd_fn = []

    for prd in period:
        for fn in [torch.sin, torch.cos]:
            embd_fn.append(lambda x, prd=prd, fn=fn: fn(prd*x))
    
    embedded = torch.cat([fn(input) for fn in embd_fn], dim=-1)

    return embedded

def sample_coarse_points(near, far, N_rays, N_samples):
    t_vals = torch.linspace(near, far, N_samples+1)
    samples = torch.rand(N_rays, N_samples) * (t_vals[1:] - t_vals[:-1])
    
    t_vals = t_vals[:-1] + samples

    return t_vals

def sample_fine_points(bins, weight, N_samples):
    weight = weight + 1e-5

    pdf = weight / torch.sum(weight, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)
    
    u = torch.rand(*cdf.shape[:-1], N_samples)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def render_rays(rays, near, far, model_coarse, model_fine, args):
    rays_o, rays_d = rays[0], rays[1]

    coarse_t = sample_coarse_points(near, far, rays_o.shape[0], args.N_samples)
    coarse_pts = rays_o[..., None, :] + coarse_t[..., None] * rays_d[..., None, :]

    delta = torch.cat([torch.norm(coarse_pts[...,1:,:]-coarse_pts[...,:-1,:], dim=-1), torch.Tensor([1e10]).expand(rays_o.shape[0], 1)], dim=-1)
    
    #coarse_pts = positional_encoding(coarse_pts, args.multires)
    r_x = positional_encoding(coarse_pts.reshape(-1, 3), 10)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:,None].expand(-1, args.N_samples,3)
    r_d = positional_encoding(viewdirs.reshape(-1, 3), 4)

    #RGB, D = get_rgbd(coarse_pts)
    out = model_coarse(r_x, r_d)
    RGB, D = out[..., :-1].reshape(-1, args.N_samples, 3), out[..., -1].reshape(-1, args.N_samples)

    RGB = torch.sigmoid(RGB)
    D = F.relu(D)

    alpha = (1. - torch.exp(-D * delta))
    weight = alpha * torch.cat([torch.ones((alpha.shape[0], 1)), torch.cumprod(1.-alpha + 1e-10, dim=-1)[...,:-1]], dim=-1)
    
    coarse_map = torch.sum(weight[...,None] * RGB, dim=-2)
    if args.white_bkgd:
        coarse_map += (1. - torch.sum(weight, -1)[..., None])

    if args.N_importance:
        fine_t = sample_fine_points(.5*(coarse_t[...,1:] + coarse_t[...,:-1]), weight[...,1:-1], args.N_importance)
        fine_t = fine_t.detach()

        total_t, _ = torch.sort(torch.cat([coarse_t, fine_t], dim=-1), dim=-1)
        total_pts = rays_o[..., None, :] + total_t[..., None] * rays_d[..., None, :]

        delta = torch.cat([torch.norm(total_pts[...,1:,:]-total_pts[...,:-1,:], dim=-1), torch.Tensor([1e10]).expand(rays_o.shape[0], 1)], dim=-1)

        r_x = positional_encoding(total_pts.reshape(-1, 3), 10)
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[:,None].expand(-1, args.N_samples + args.N_importance,3)
        r_d = positional_encoding(viewdirs.reshape(-1, 3), 4)
        
        out = model_fine(r_x, r_d)
        RGB, D = out[..., :-1].reshape(-1, args.N_importance + args.N_samples, 3), out[..., -1].reshape(-1, args.N_importance + args.N_samples)

        RGB = torch.sigmoid(RGB)
        D = F.relu(D)

        alpha = 1 - torch.exp(-D * delta)
        weight = alpha * torch.cat([torch.ones(alpha.shape[0], 1), torch.cumprod(1.-alpha + 1e-10, dim=-1)[...,:-1]], dim=-1)
        fine_map = torch.sum(weight[...,None] * RGB, dim=-2)

        if args.white_bkgd:
            fine_map += (1. - torch.sum(weight, -1)[..., None])
        
        return coarse_map, fine_map
    

    return coarse_map, None # N_rays * [R, G, B]


def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H))

    i = i.t()
    j = j.t()

    dirs = torch.stack([(i-K[0][2]) / K[0][0], -(j-K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')

    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))

    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    o_x = -focal * (W/2) * (rays_o[..., 0] / rays_o[..., 2])
    o_y = -focal * (H/2) * (rays_o[..., 1] / rays_o[..., 2])
    o_z = 1 + 2 * near / rays_o[..., 2]

    ndc_o = torch.stack([o_x, o_y, o_z], dim=-1)

    d_x = -focal * (W/2) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d_y = -focal * (H/2) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d_z = -2 * near / rays_o[..., 2]

    ndc_d = torch.stack([d_x, d_y, d_z], dim=-1)

    return ndc_o, ndc_d

if __name__ == '__main__':

    from load_blender import load_blender_data
    import os
    import imageio
    
    basedir =  '/home/khc/nerf-pytorch/data/nerf_synthetic/lego'

    imgs, poses, render_poses, [H, W, focal], i_split = load_blender_data(basedir, 8)
    near = 2. #hyper param
    far = 6. #hyper param

    train_idx, test_idx, val_idx = i_split
    K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    vid = []
    for i, pose in enumerate(render_poses):
        ray_o, ray_d = map(torch.Tensor, get_rays_np(H, W, K, pose))
        ray_o = ray_o.view(-1,3)
        ray_d = ray_d.view(-1,3)
        ray_d = torch.norm(ray_d, dim=-1)[...,None] * ray_d
        rgbs = render_rays(torch.cat([ray_o, ray_d], dim=-1), H, W, focal, near, far, 64).numpy()
        vid.append(rgbs)

        filename = f'/home/khc/nerf/test/test_{i}.png'
        imageio.imwrite(filename, to8b(rgbs))
    
    vid = np.stack(vid, 0)
    imageio.mimwrite('/home/khc/nerf/test/rgb.mp4', to8b(vid), fps=30, quality=8)