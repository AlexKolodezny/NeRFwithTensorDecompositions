import random
from warnings import warn
import numpy as np
from itertools import cycle
import math

import imageio
from tqdm import tqdm, trange
import wandb

import time

from .helpers import *
from .load_blender import load_blender_data
from .load_tnt import load_tnt_data
from .radiance_field import RadianceField
from ..models.spherical_harmonics import spherical_harmonics_bases

def create_model(args):
    model = RadianceField(args)
    print(model)

    if args.optimizer == "Adam":
        optimizers = [torch.optim.Adam(
            params=model.get_param_groups(),
            lr=args.lrate,
            betas=(0.9, 0.999),
        )]
    elif args.optimizer == "SGD":
        optimizers = [torch.optim.SGD(
            params=model.get_param_groups(),
            lr=args.lrate,
        )]
    elif args.optimizer == "SeparatedSGD":
        optimizers = [
            torch.optim.SGD(
                params=[{"params": [param], "tag": group["tag"], "lr": group["lr"]}],
                lr=args.lrate,
            )
            for group in model.get_param_groups()
            for param in group["params"]
        ]
    elif args.optimizer == "LBFGS":
        optimizers = [torch.optim.LBFGS(
            params=model.parameters(),
            lr=args.lrate,
            history_size=args.history_size,
            max_iter=args.lbfgs_iters,
        )]
    elif args.optimizer == "SeparatedLBFGS":
        optimizers = [
            torch.optim.LBFGS(
                params=[group],
                lr=args.lrate,
                line_search_fn = 'strong_wolfe',
                max_eval=max(20, args.lbfgs_iters * 1.25),
                max_iter=args.lbfgs_iters,
                history_size=1,
            )
            for group in model.get_param_groups()
        ]



    if args.parallel:
        model = torch.nn.DataParallel(model).to(args.device)
    else:
        model = model.to(args.device)

    render_kwargs_train = {
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'model': model,
        'use_viewdirs': args.use_viewdirs,
        'dim_grid': args.dim_grid,
        'adjust_near_far': args.adjust_near_far,
        'filter_rays': args.filter_rays,
        'dir_center_pix': args.dir_center_pix,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'expand_pdf_value': args.expand_pdf_value,
        'checks': args.checks,
        'lossfn': args.lossfn,
        'sigma_activation': args.sigma_activation,
        'alpha': args.alpha,
        'dist_constant': args.dist_constant,
        'use_new_march_rays': args.use_new_march_rays,
        'sample_points': args.sample_points,
        'use_EM': args.use_EM,
        'perturb_direction': args.perturb_direction,
    }

    print('Not ndc!')
    render_kwargs_train['ndc'] = False

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = 0
    render_kwargs_test['perturb_direction'] = 0
    render_kwargs_test['raw_noise_std'] = 0.

    return model, render_kwargs_train, render_kwargs_test, optimizers


def march_rays(
        raw_rgb,
        raw_sigma,
        z_vals,
        targets,
        raw_noise_std=0.,
        white_bkgd=False,
        ret_weights=False,
        use_new_integration=True,
        sigma_activation=None,
        lossfn=None,
):
    NR, NS = raw_sigma.shape

    if raw_noise_std > 0.:
        raw_sigma += torch.randn(NR, NS, device=raw_sigma.device) * raw_noise_std  # NR x NS

    dists = z_vals[..., 1:] - z_vals[..., :-1]  # NR x (NS-1)

    if sigma_activation == 'relu':
        sigma = F.relu(raw_sigma)  # NR x NS
        integrate_fn = integrate_new if use_new_integration else integrate_old
        weights = integrate_fn(sigma, dists)  # NR x NS
    elif sigma_activation == "logloss":
        sigma = F.softplus(raw_sigma)
        integrate_fn = integrate_new if use_new_integration else integrate_old
        weights = integrate_fn(sigma, dists)  # NR x NS
        
    weights = torch.cat([weights, 1. - weights.sum(dim=1)[:,None]], dim=1)

    rgb_map = torch.sum(weights[..., None] * raw_rgb, -2)  # [N_rays, 3]

    out = {'rgb_map': rgb_map.detach()}
    if ret_weights:
        out['weights'] = weights.detach()

    if targets is not None:
        img_loss = {
            'huber': F.huber_loss,
            'mse': F.mse_loss,
        }[lossfn](rgb_map, targets)
        out['losses'] = img_loss[None].detach()

        img_loss.backward()

    return out


def render_rays(
        ray_batch,
        target_batch,
        model,
        N_samples,
        update_statistics=[],
        calc_elbo_for_new=[],
        use_viewdirs=False,
        dim_grid=None,
        perturb=0,
        perturb_direction=0,
        N_importance=0,
        white_bkgd=False,
        raw_noise_std=0.,
        expand_pdf_value=None,
        sigma_warmup_sts=False,
        sigma_warmup_numsteps=None,
        cur_step=None,
        checks=True,
        lossfn=None,
        alpha=None,
        use_EM=0,
        sigma_activation='relu',
        sample_points='adjusted',
        dist_constant=False,
        use_new_march_rays=False,
):
    """
    Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      model: function. Model for predicting RGB and density at each point
        in space.
      N_samples: int. Number of different times to sample along each ray.
      perturb: int, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      expand_pdf_value: Optional[float]. When not None, use MIP-NERF weight filter
        with the given padding.
      checks: bool. Extended asserts.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
    """

    if use_new_march_rays:
        rgb_map, weights = model.render_rays(ray_batch, N_samples, cur_step=cur_step)
        out = {'rgb_map': rgb_map.detach()}

        if target_batch is not None:
            img_loss = {
                'huber': F.huber_loss,
                'mse': F.mse_loss,
            }[lossfn](rgb_map, target_batch)
            img_loss.backward()

            out['losses'] = img_loss[None].detach()
        return out


    def sample_rays(inputs, viewdirs):
        """
        Prepares inputs and applies network 'fn'.
        :param inputs (torch.Tensor): sampled points of shape [batch x ray x 3]
        :param viewdirs (torch.Tensor): directions corresponding to inputs rays of shape [batch x 3]
        """
        rgb, sigma, std, mask = model(inputs, viewdirs)
        if sigma_warmup_sts and cur_step <= sigma_warmup_numsteps:
            sigma *= cur_step / sigma_warmup_numsteps
        return rgb, sigma, std, mask

    N_rays, device = ray_batch.shape[0], ray_batch.device
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    near, far = ray_batch[:, 6:7], ray_batch[:, 7:8]
    radii = ray_batch[:,8:9]
    rays_d_unnorm = ray_batch[:,9:12]
    rays_dx = ray_batch[:,12:15]
    rays_dy = ray_batch[:,15:18]

    if perturb_direction:
        alpha = torch.normal(torch.zeros(N_rays, 2, device=device), 0.5 / 12**0.5)
        rays_d_unnorm = alpha[:,:1] * rays_dx + alpha[:,1:] * rays_dy + rays_d_unnorm
        rays_d = rays_d_unnorm / rays_d_unnorm.norm(dim=1)[:,None]

    viewdirs = None
    if use_viewdirs:
        viewdirs = rays_d.clone()

    valid_mask = None
    if sample_points == 'adjusted':
        t_far = torch.linspace(0., 1.0, steps=N_samples, device=device)
        t_near = 1.0 - t_far
        z_vals = near * t_near + far * t_far
        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    elif sample_points == 'equal':
        z_vals = near + 2 * 3**0.5 / N_samples * torch.arange(N_samples + 1, device=device)
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        dists = mids[..., 1:] - mids[..., :-1]  # NR x (NS-1)
        if perturb:
            upper = z_vals[..., 1:]
            lower = z_vals[..., :-1]
            t_rand = torch.rand(mids.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand
        else:
            z_vals = mids
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    elif sample_points == 'grid':
        z_vals, valid_mask = model.get_intersect_coords(rays_o, rays_d)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    raw_rgb, raw_sigma, raw_var, mask = sample_rays(pts, viewdirs)
    if valid_mask is None:
        valid_mask = mask

    z_vals *= (dim_grid - 1) * 0.5
    out = march_rays(
        raw_rgb,
        raw_sigma,
        z_vals,
        target_batch,
        raw_noise_std,
        white_bkgd,
        lossfn=lossfn,
        ret_weights=N_importance > 0,
        sigma_activation=sigma_activation)

    return out


def render_rays_chunks(rays_flat, chunk, targets, **kwargs):
    all_ret = {}
    if chunk >= rays_flat.shape[0]:
        return render_rays(rays_flat, targets, **kwargs)

    for i in trange(0, rays_flat.shape[0], chunk, disable=True):
        target_batch = targets[i:i+chunk] if targets is not None else None
        ret = render_rays(rays_flat[i:i + chunk], target_batch, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(
        H, W, K, chunk, rays=None, targets=None, update_statistics=None, c2w=None, ndc=True, near=None, far=None,
        calc_elbo_for_new=None,
        adjust_near_far=False, filter_rays=False, dir_center_pix=True,
        sigma_warmup_sts=False,
        sigma_warmup_numsteps=None,
        cur_step=None,
        **kwargs
):
    """
    Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      K: Tensor. Camera intrinsics matrix
      chunk: int. Maximum number of rays to process simultaneously. Used to control maximum memory usage.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for each example in batch.
      targets: TODO,
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float. Nearest distance for a ray.
      far: float. Farthest distance for a ray.
      adjust_near_far: bool. If True, computes per-pixel near and far values from intersection with the AABB.
      filter_rays: bool. If True, skips rays not passing through AABB.
      dir_center_pix: bool. If True, offsets ray directions to pass through voxel centers instead of corners.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      extras: dict with everything returned by render_rays().
      loss: loss for batch
    """
    near_vec, far_vec, valid_mask = None, None, None
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d, near_vec, far_vec, valid_mask, radii, rays_d_unnorm, rays_dx, rays_dy = get_rays(
            H, W, K, c2w, dir_center_pix=dir_center_pix, valid_only=filter_rays
        )
    else:
        # use provided ray batch
        if len(rays) == 4:
            rays_o, rays_d, radii, rays_d_unnorm, rays_dx, rays_dy = rays
        else:
            rays_o, rays_d, near_vec, far_vec, radii, rays_d_unnorm, rays_dx, rays_dy = rays

    if adjust_near_far:
        near, far = near_vec, far_vec
    else:
        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    rays = torch.cat([rays_o, rays_d, near, far, radii[:,None], rays_d_unnorm, rays_dx, rays_dy], -1)

    # Render
    all_ret = render_rays_chunks(
        rays, chunk, targets,
        sigma_warmup_sts=sigma_warmup_sts,
        sigma_warmup_numsteps=sigma_warmup_numsteps,
        cur_step=cur_step,
        update_statistics=update_statistics,
        calc_elbo_for_new=calc_elbo_for_new,
        **kwargs
    )

    if c2w is not None:
        # recover image shapes
        for k, v in all_ret.items():
            if filter_rays:
                fill_value = 1.0 if kwargs['white_bkgd'] and k == 'rgb_map' else 0.0
                out_v = torch.full((H, W, *v.shape[1:]), fill_value=fill_value, dtype=v.dtype, device=v.device)
                out_v[valid_mask] = v
            else:
                out_v = v.reshape(H, W, *v.shape[1:])
            all_ret[k] = out_v

    return all_ret


@torch.no_grad()
def render_path(
        render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0,
        compute_stats=False, desc=None, device=None
):
    H, W, _ = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor

    rgbs = []

    psnr, mse, ssim, lpips = 0., 0., 0., 0.

    for i, c2w in enumerate(tqdm(render_poses, desc=desc)):
        out = render(H, W, K, chunk=chunk, c2w=c2w.to(device), **render_kwargs)
        rgb_pred_cuda = out['rgb_map']
        rgb_pred_np = rgb_pred_cuda.cpu().numpy()
        rgbs.append(rgb_pred_np)

        if compute_stats:
            assert gt_imgs is not None
            assert render_factor == 0
            rgb_gt_np = gt_imgs[i]
            rgb_gt_cuda = torch.from_numpy(rgb_gt_np).to(rgb_pred_cuda)
            mse_i = F.mse_loss(rgb_pred_cuda, rgb_gt_cuda)
            mse += mse_i
            psnr += -10. * mse_i.log10()
            ssim += rgb_ssim(rgb_pred_np, rgb_gt_np, 1)
            lpips += rgb_lpips(rgb_pred_cuda, rgb_gt_cuda)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)

    if compute_stats:
        mse /= len(render_poses)
        psnr /= len(render_poses)
        ssim /= len(render_poses)
        lpips /= len(render_poses)
        return rgbs, psnr, mse, ssim, lpips

    return rgbs, psnr, mse


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--seed", type=int, default=2022,
                        help='RNG seed')
    parser.add_argument("--log_root", type=str, required=True,
                        help='log root path, where each experiment logs into its named subdirectory (see expname)')
    parser.add_argument("--dataset_root", type=str, required=True,
                        help='input data root, containing relevant datasets in subdirectories')
    parser.add_argument("--dataset_type", type=str, default='blender',
                        help='options: blender')
    parser.add_argument("--dataset_dir", type=str,
                        help='a subdirectory within the specified dataset type')
    parser.add_argument("--unbounded", type=int, default=0)

    parser.add_argument("--device", type=str, default='cpu',
                        help='device to use')
    parser.add_argument("--parallel", action='store_true',
                        help='use DataParallel')

    # voxel grid configuration
    parser.add_argument("--models", type=str,
                        help='models of voxel frid in json')
    parser.add_argument("--model", type=str, default="QTTNF", choices=("QTTNF", "TackerNF", "VMNF", "SkeletonNF"),
                        help='model of voxel frid')
    parser.add_argument("--dim_grid", type=int, default=256,
                        help='size of voxel grid')
    parser.add_argument("--fine_dim_grid", type=int, default=256)
    parser.add_argument("--init_method", type=str, default='normal', choices=('normal', 'zeros', 'eye'),
                        help='voxel grid initialization')
    parser.add_argument("--grid_type", type=str, default='fused', choices=('fused', 'separate'),
                        help='type of voxel grid compression')
    parser.add_argument("--use_rgb_sigmoid", type=int, default=1,
                        help='use sigmoid on rgb')
    parser.add_argument("--sigma_activation", type=str, default='relu', choices=('relu', 'logloss'),
                        help='sigma activation')
    parser.add_argument("--dist_constant", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--use_new_march_rays", type=int, default=0)
    parser.add_argument("--sample_points", type=str, default='adjusted', choices=('equal', 'adjusted', 'grid', 'ndc'))
    parser.add_argument("--use_EM", type=int, default=0)
    parser.add_argument("--normalize", type=int, default=0)
    parser.add_argument("--intersect_threshold", type=float, default=1e-2)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--perturb_direction", type=int, default=0)
    
    parser.add_argument("--sigma_warmup_sts", type=int, default=1,
                        help='sigma warmup at the beginning of training')
    parser.add_argument("--sigma_warmup_numsteps", type=int, default=1000,
                        help='sigma warmup duration in steps')

    parser.add_argument("--dtype", type=str, default='float32', choices=('float16', 'float32', 'float64'),
                        help='voxel grid dtype')
    parser.add_argument("--checks", type=int, default=1,
                        help='performs expensive sanity checks before addressing voxels grid')

    # training options
    parser.add_argument("--N_iters", type=int, default=200000,
                        help='number of training iterations')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_sigma_multiplier", type=float, default=1.0,
                        help='learning rate multiplier for sigma (applied to lrate value)')
    parser.add_argument("--lrate_shader", type=float, default=0.001,
                        help='learning rate of MLP shader')
    parser.add_argument("--lrate_decay", type=int, default=250000,
                        help='number of steps after which LR is decayed by 0.1')
    parser.add_argument("--lrate_warmup_steps", type=int, default=0,
                        help='number of steps to warmup LR in the beginning')
    parser.add_argument("--chunk", type=int, default=4096,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--load_model", type=str, default=None,
                        help='specific weights npy file to initialize model weights')
    parser.add_argument("--init_model_ttsvd", type=str, default=None,
                        help='weights of a full voxel grid of compatible configuration')
    parser.add_argument("--lossfn", type=str, default='huber', choices=('mse', 'huber', 'lrf'),
                        help='loss function to use during training')
    parser.add_argument("--train_size", type=int, default=None,
                        help='number of rays for train')
    parser.add_argument("--train_images", nargs="+", type=int, default=None)
    parser.add_argument("--log_train_images", nargs="+", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=('Adam', 'SGD', 'SeparatedSGD', 'LBFGS', 'SeparatedLBFGS'),
                        help='specify optimizer')
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--lbfgs_iters", type=int, default=20,
                        help='iterations in lbfgs')
    parser.add_argument("--weight_threshold", type=float, default=None)
    parser.add_argument("--use_lrate_decay", type=int, default=True)
    parser.add_argument("--history_size", type=int, default=10)
    parser.add_argument("--clear_cache", type=int, default=0)

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=int, default=1,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--filter_rays", type=int, default=1,
                        help='reject rays that do not intersect the grid')
    parser.add_argument("--adjust_near_far", type=int, default=1,
                        help='try to adjust near and far for each ray')
    parser.add_argument("--use_viewdirs", type=int, default=1,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--shading_mode", type=str, default='spherical_harmonics',
                        choices=('spherical_harmonics', 'mlp'),
                        help='Select shading mode for RGB values')
    parser.add_argument("--sh_basis_dim", type=int, default=9,
                        help='spherical harmonics basis dimension per channel')
    parser.add_argument("--fine_sh_basis_dim", type=int, default=9)
    parser.add_argument("--rgb_feature_dim", type=int, default=27,
                        help='RGB feature vector dimension')
    parser.add_argument("--dir_center_pix", type=int, default=1,
                        help='pass rays through voxel center instead of corner')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--expand_pdf_value", type=float, default=0.01,
                        help='When not None (default), uses MIP-NeRF weight filtering with the given padding value')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # other dataset options
    parser.add_argument("--image_downscale_factor", type=int, default=1,
                        help='downscale images (and poses) factor')
    parser.add_argument("--image_downscale_filter", type=str, default='antialias', choices=('area', 'antialias'),
                        help='downscale images filter')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--scene_scale", type=float, default=1.0,
                        help='3D scene scaling to better fit voxel grid')
    parser.add_argument("--scene_rot_z_deg", type=int, default=0,
                        help='3D scene rotation around z axis')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    # logging/saving options
    parser.add_argument("--i_wandb", type=int, default=1,
                        help='output progress via wandb')
    parser.add_argument("--i_tqdm", type=int, default=1,
                        help='output progress via tqdm')
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_img_ids", nargs="+", type=int, default=[0],
                        help='image ids to output periodically to tensorboard')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():
    torch.autograd.set_detect_anomaly(True)

    parser = config_parser()
    args, extras = parser.parse_known_args()
    if len(extras) > 0:
        warn('Unknown arguments: ' + str(extras))

    log_dir = os.path.join(args.log_root, args.expname)
    os.makedirs(log_dir, exist_ok=True)

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # check the experiment is not resumed with different code and or settings
    verify_experiment_integrity(args)

    # if args.i_wandb:
    wandb.init(
        project='NeRFwithTensorDecompositions',
        config=args,
        name=args.expname,
        # dir=log_dir,
        # force=True,  # makes the user provide wandb online credentials instead of running offline
    )
        # wandb.tensorboard.patch(
        #     save=False,  # copies tb files into cloud and allows to run tensorboard in the cloud
        #     pytorch=True,
        # )

    # Load data
    dataset_path = os.path.join(args.dataset_root, {
        'blender': 'nerf_synthetic',
        'tnt': 'tanks_and_temples',
    }[args.dataset_type], args.dataset_dir)
    K = None
    if args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            dataset_path,
            args.image_downscale_factor,
            args.image_downscale_filter,
            args.testskip,
            args.scene_scale,
            args.scene_rot_z_deg,
        )
        print('Loaded blender', images.shape, render_poses.shape, hwf, dataset_path)

        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
    elif args.dataset_type == "tnt":
        images, poses, render_poses, hwf, i_split = load_tnt_data(
            dataset_path,
            args.image_downscale_factor,
            args.image_downscale_filter,
            args.testskip,
            args.scene_scale,
            args.scene_rot_z_deg,
        )
        print('Loaded tanks and temples', images.shape, render_poses.shape, hwf, dataset_path)
        i_train, i_test = i_split
        i_val = None

        near = 2.
        far = 6.
    else:
        raise RuntimeError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    f = os.path.join(log_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(log_dir, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create model
    model, render_kwargs_train, render_kwargs_test, optimizers = create_model(args)
    global_step = 0

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        model.eval()
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(
            log_dir, 'renderonly_{}'.format('test' if args.render_test else 'path')
        )
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, _, psnr, _, ssim, lpips = render_path(
            render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
            savedir=testsavedir, render_factor=args.render_factor, compute_stats=True,
            desc='renderonly', device=args.device)
        print('Done rendering', testsavedir, 'PSNR:', psnr, 'SSIM:', ssim, 'LPIPS:', lpips)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

        return

    # precompute all rays
    poses = torch.Tensor(poses)

    ds_rays_o, ds_rays_d, ds_near, ds_far, ds_target = [], [], [], [], []
    ds_rays_radii, ds_rays_d_unnorm = [], []
    ds_rays_dx, ds_rays_dy = [], []
    if args.train_images is not None:
        i_train = args.train_images
    for img_i in i_train:
        target = images[img_i]
        target = torch.Tensor(target)
        pose = poses[img_i, :3, :4]

        rays_o, rays_d, near, far, valid_mask, radii, rays_d_unnorm, rays_dx, rays_dy = get_rays(
            H, W, K, pose, dir_center_pix=args.dir_center_pix, valid_only=args.filter_rays,
        )

        if args.filter_rays:
            target = target[valid_mask]
        else:
            target = target.view(-1, 3)

        ds_rays_o.append(rays_o)
        ds_rays_d.append(rays_d)
        ds_near.append(near)
        ds_far.append(far)
        ds_target.append(target)
        ds_rays_radii.append(radii)
        ds_rays_d_unnorm.append(rays_d_unnorm)
        ds_rays_dx.append(rays_dx)
        ds_rays_dy.append(rays_dy)
    
    ds_rays_o = torch.cat(ds_rays_o, dim=0)
    ds_rays_d = torch.cat(ds_rays_d, dim=0)
    ds_near = torch.cat(ds_near, dim=0)
    ds_far = torch.cat(ds_far, dim=0)
    ds_target = torch.cat(ds_target, dim=0)
    ds_rays_radii = torch.cat(ds_rays_radii, dim=0)
    ds_rays_d_unnorm = torch.cat(ds_rays_d_unnorm, dim=0)
    ds_rays_dx = torch.cat(ds_rays_dx, dim=0)
    ds_rays_dy = torch.cat(ds_rays_dy, dim=0)

    if args.normalize:
        model.target_mean = ds_target.mean()
        model.target_std = ds_target.std()
        print("MEAN: ", model.target_mean.item())
        print("STD: ", model.target_std.item())
    else:
        model.target_mean = 0
        model.target_std = 1

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    wandb.log({
        'num_uncompressed_params': model.num_uncompressed_params,
        'num_compressed_params': model.num_compressed_params,
        # 'sz_uncompressed_gb': model.sz_uncompressed_gb,
        # 'sz_compressed_gb': model.sz_compressed_gb,
        # 'compression_factor': model.compression_factor,
    }, step=0)

    if args.train_size is not None:
        id_sampler = iter(ScramblingSampler(
            np.random.permutation(ds_target.shape[0])[:args.train_size],
            args.N_rand))
    else:
        id_sampler = iter(ScramblingSampler(ds_target.shape[0], args.N_rand))
    
    model.train()
    j = 0
    for i, optimizer in zip(trange(0, args.N_iters + 1, disable=not args.i_tqdm), cycle(optimizers)):
        ids = next(id_sampler)
        rays_o = ds_rays_o[ids].to(args.device)
        rays_d = ds_rays_d[ids].to(args.device)
        target = ds_target[ids].to(args.device)
        rays_d_unnorm = ds_rays_d_unnorm[ids].to(args.device)
        radii = ds_rays_radii[ids].to(args.device)
        rays_dx = ds_rays_dx[ids].to(args.device)
        rays_dy = ds_rays_dy[ids].to(args.device)
        if args.adjust_near_far:
            near = ds_near[ids].to(args.device)
            far = ds_far[ids].to(args.device)
            batch_rays = [rays_o, rays_d, near, far, radii, rays_d_unnorm, rays_dx, rays_dy]
        else:
            batch_rays = [rays_o, rays_d, radii, rays_d_unnorm, rays_dx, rays_dy]

        mse_loss = None
        loss = None

        def closure():
            nonlocal j
            j += 1

            optimizer.zero_grad()

            out = render(
                H, W, K,
                chunk=args.chunk,
                rays=batch_rays,
                targets=target,
                sigma_warmup_sts=args.sigma_warmup_sts,
                sigma_warmup_numsteps=args.sigma_warmup_numsteps,
                cur_step=j-1,
                **render_kwargs_train
            )

            if j % args.i_print == 0:
                nonlocal mse_loss
                nonlocal loss
                mse_loss = F.mse_loss(out['rgb_map'], target).detach().cpu()
                loss = out['losses'].mean().detach().cpu()

                psnr = mse2psnr(mse_loss)
                val = {
                    "iter": j,
                    "loss": loss.item(),
                    "mse_loss": mse_loss.item(),
                    "psrn": psnr.item(),
                }
                if "elbo" in out:
                    print("ELBO: ", out['elbo'].sum(), i)
                    val["elbo"] = out['elbo'].sum()
                msg = f"[TRAIN] Iter: {j} of {args.N_iters}, loss: {loss.item()}"
                msg += f', PSNR: {psnr.item()}'
                wandb.log(val, step=j)
                print(msg)

            return out['losses'].sum().item()

        if args.optimizer in {"LBFGS", "SeparatedLBFGS"}:
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()
        if args.clear_cache:
            torch.cuda.empty_cache()
    
        if args.use_lrate_decay:
            decay_rate = 0.1
            new_lrate = args.lrate * (decay_rate ** (global_step / args.lrate_decay))
            if args.lrate_warmup_steps > 0 and global_step < args.lrate_warmup_steps:
                new_lrate *= global_step / args.lrate_warmup_steps
            for param_group in optimizer.param_groups:
                if param_group['tag'] == 'vox':
                    param_group['lr'] = new_lrate

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(log_dir, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)

            trainsavedir = os.path.join(log_dir, 'trainset_{:06d}'.format(i))
            os.makedirs(trainsavedir, exist_ok=True)

            model.eval()
            _, test_psnr, test_mse, test_ssim, test_lpips = render_path(
                torch.Tensor(poses[i_test]), hwf, K, args.chunk, render_kwargs_test,
                gt_imgs=images[i_test], savedir=testsavedir, compute_stats=True, desc='test set', device=args.device
            )
            if args.log_train_images is not None:
                _, train_psnr, train_mse, train_ssim, train_lpips = render_path(
                    torch.Tensor(poses[args.log_train_images]), hwf, K, args.chunk, render_kwargs_train,
                    gt_imgs=images[args.log_train_images], savedir=trainsavedir, compute_stats=True, desc='train set', device=args.device
                )
            model.train()

            wandb.log({
                'test_mse': test_mse,
                'test_psnr': test_psnr,
                'test_ssim': test_ssim,
                'test_lpips': test_lpips,
            }, step=i)

            # wandb.log({
            #     'train_mse': train_mse,
            #     'train_psnr': train_psnr,
            #     'train_ssim': train_ssim,
            #     'train_lpips': train_lpips,
            # }, step=i)

            tqdm.write(f"[TEST] Iter: {i} of {args.N_iters}, PSNR: {test_psnr}, SSIM: {test_ssim}, LPIPS: {test_lpips}")
            tqdm.write(f"[TRAIN] Iter: {i} of {args.N_iters}, PSNR: {train_psnr}, SSIM: {train_ssim}, LPIPS: {train_lpips}")

        global_step += 1

    path = os.path.join(log_dir, 'final.pth')
    torch.save(render_kwargs_train['model'].state_dict(), path)
    tqdm.write(f'Saved model: {path}')
    wandb.finish()


if __name__ == '__main__':
    train()