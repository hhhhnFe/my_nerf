import os
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from load_blender import load_blender_data
from run_nerf_helpers import *
from nerf_class import *

def config_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser()

    #parser.add_argument('--config', is_config_file=True, 
    #                   help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/nerf_synthetic/lego', 
                        help='input data directory')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    
    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N imgs from test/val sets, useful for large datasets like deepvoxels')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF imgs')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N imgs as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    # Parallel test
    parser.add_argument("--gpu_num", type=int, default=0,
                        help='Number of device')

    return parser.parse_args()

def train(args, H, W, iters, train_data, temp_rays, model_coarse, model_fine):

    learnable_params = list(model_coarse.parameters())
    if model_fine != None:
        learnable_params += list(model_fine.parameters())
    optimizer = torch.optim.Adam(params=learnable_params, lr=args.lrate, betas=(0.9, 0.999))

    global_step = 0
    i_batch = 0
    if args.no_batching:
        chosen = train_data[np.random.choice(range(train_data.shape[0]))]
        if global_step < args.precrop_iters:
            p = args.precrop_frac
            chosen = chosen[int((1.-p) * (H//2)):int((1.+p) * (H//2))][int((1.-p) * (W//2)):int((1.+p) * (W//2))]
        
        chosen = torch.Tensor(np.reshape(chosen, [-1,3,3]))
        bound = chosen.shape[0]
        chosen = chosen[torch.randperm(bound)]

    else: bound = train_data.shape[0]

    for iter in trange(iters):
        if args.no_batching:
            if i_batch >= bound:
                chosen = train_data[np.random.choice(range(train_data.shape[0]))]
                if iter < args.precrop_iters:
                    chosen = chosen[int((1.-p) * (H//2)):int((1.+p) * (H//2))][int((1.-p) * (W//2)):int((1.+p) * (W//2))]
                chosen = torch.Tensor(np.reshape(chosen, [-1,3,3]))
                bound = chosen.shape[0]

                chosen = chosen[torch.randperm(bound)]
                i_batch = 0

            batch = chosen[i_batch:i_batch+args.N_rand]
        else:
            if i_batch >= bound:
                print("Shuffle data after an epoch")
                mixed = torch.randperm(bound)
                train_data = train_data[mixed]
                i_batch = 0

            batch = train_data[i_batch:i_batch+args.N_rand]

        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_rgb = batch[:2], batch[2]   # (ray_o, ray_d) * N_rand * 3, N_rand * 3
        i_batch += args.N_rand

        coarse_out, fine_out = render_rays(batch_rays, 2., 6., model_coarse, model_fine, args)

        loss = img2mse(coarse_out, target_rgb)
        if fine_out != None:
            loss_c = img2mse(fine_out, target_rgb)
            psnr = mse2psnr(loss_c)
            loss += loss_c
        else: psnr = mse2psnr(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if iter % args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {iter} Loss: {loss.item()}  PSNR: {psnr.item()}")
        if iter % 10000 == 0 and iter != 0:
            with torch.no_grad():
                print('Render temp view')
                if model_fine != None:
                    temp_view = torch.stack([render_rays(temp_rays[:,i:i+4000], 2., 6., model_coarse, model_fine, args)[1] for i in range(0, temp_rays[0].shape[0], 4000)])
                else:
                    temp_view = torch.stack([render_rays(temp_rays[:,i:i+4000], 2., 6., model_coarse, model_fine, args)[0] for i in range(0, temp_rays[0].shape[0], 4000)])
                temp_view = temp_view.view(H, W, 3).cpu()
                temp_view = temp_view.numpy()
                filename = os.path.join(args.basedir, args.expname, f'GPU:{args.gpu_num}_test_{iter}.png')
                imageio.imwrite(filename, to8b(temp_view))

        global_step += 1

if __name__ == '__main__':
    args = config_parser()

    if torch.cuda.is_available():
        if args.gpu_num < torch.cuda.device_count():
            torch.cuda.set_device(args.gpu_num)
            device = torch.device(f'cuda:{args.gpu_num}')
        else:
            device = 'cuda'
    else:
        device = 'cpu'
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset_type == 'blender':
        imgs, poses, render_poses, [H, W, focal], i_split = load_blender_data(args.datadir, args.testskip, args.half_res)
        near = 2. #hyper param
        far = 6. #hyper param
        train_idx, test_idx, val_idx = i_split

        if args.white_bkgd:
            imgs = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
        else:
            imgs = imgs[...,:3]
        

    K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    train_rays = np.stack([get_rays_np(H, W, K, poses[i]) for i in train_idx]) # N * 2(=[ray_o, ray_d]) * H * W * 3
    train_imgs = np.stack([imgs[i] for i in train_idx]) # N * H * W * 3
    train_data = np.concatenate([train_rays, train_imgs[:, np.newaxis]], 1) # (N_train_data) * 3(=[ray_o, ray_d, RGB]) * H * W * 3 
    train_data = np.transpose(train_data, [0,2,3,1,4]) # N * H * W * 3(ray_o, ray_d, RGB) * 3
    train_data = train_data.astype(np.float32)

    if not args.no_batching:
        train_data = np.reshape(train_data, [-1,3,3]) # (N*H*W) * 3(ray_o, ray_d, RGB) * 3
        np.random.shuffle(train_data)
        print("Shuffled")
        train_data = torch.Tensor(train_data).to(device)

    ray_o, ray_d = map(torch.Tensor, get_rays_np(H, W, K, render_poses[0]))
    ray_o = ray_o.view(-1,3)
    ray_d = ray_d.view(-1,3)
    ray_o = ray_o.to(device)
    ray_d = ray_d.to(device)

    model_coarse = NeRF().to(device)
    if args.N_importance:
        model_fine = NeRF().to(device)
    else: model_fine = None
    
    
    print("Train start")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train(args, H, W, 200000, train_data, torch.stack([ray_o, ray_d]), model_coarse, model_fine)
    
    with torch.no_grad():
        print('Final rendering')
        vid = []
        for i, pose in enumerate(render_poses):
            ray_o, ray_d = map(torch.Tensor, get_rays_np(H, W, K, pose))
            ray_o = ray_o.view(-1,3)
            ray_d = ray_d.view(-1,3)
            rgbs = torch.stack([render_rays(torch.stack([ray_o, ray_d])[:,i:i+4000], 2., 6., model_coarse, model_fine, args)[1] for i in range(0, ray_o.shape[0], 4000)])
            rgbs = rgbs.view(H, W, 3).cpu()
            rgbs = rgbs.numpy()
            vid.append(rgbs)
            filename = os.path.join(args.basedir, args.expname, f'GPU:{args.gpu_num}_final_{i}.png')
            imageio.imwrite(filename, to8b(rgbs))
    
        vid = np.stack(vid, 0)
        vidname = os.path.join(args.basedir, args.expname, f'GPU:{args.gpu_num}_final_.mp4')
        imageio.mimwrite(vidname, to8b(vid), fps=30, quality=8)
        
