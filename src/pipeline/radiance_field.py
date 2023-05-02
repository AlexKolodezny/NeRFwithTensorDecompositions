import math

import torch
import torch.nn.functional as F
import json
from copy import deepcopy

from .helpers import positional_encoding, integrate_new
from ..models.qttnf2 import QTTNF
from ..models.tacker import TackerNF
from ..models.vm import VMNF
from ..models.skeleton import SkeletonNF
from ..models.skeleton_v2 import SkeletonV2NF
from ..models.full import FullNF
from ..models.full import FullNFForRGB
from ..models.full import FullNFForSigmaBetaDist
from ..models.full import FullNFForSigmaExpDist
from ..models.full import FullNFForVar
from ..models.ttnf2 import TTNF
from ..models.ttcp import TTCPNF
from ..models.tttacker import TTTackerNF
from ..models.vttm import VTTMNF
from ..models.kronecker import KroneckerNF
from ..models.cp import CPNF

from ..models.spherical_harmonics import spherical_harmonics_bases

model_dict = {
    "QTTNF": QTTNF,
    "TTNF": TTNF,
    "TackerNF": TackerNF,
    "VMNF": VMNF,
    "SkeletonNF": SkeletonNF,
    "SkeletonV2NF": SkeletonV2NF,
    "FullNF": FullNF,
    "FullNFForRGB": FullNFForRGB,
    "FullNFForSigmaBetaDist": FullNFForSigmaBetaDist,
    "FullNFForSigmaExpDist": FullNFForSigmaExpDist,
    "FullNFForVar": FullNFForVar,
    "TTCPNF": TTCPNF,
    "TTTackerNF": TTTackerNF,
    "VTTMNF": VTTMNF,
    "KroneckerNF": KroneckerNF,
    "CPNF": CPNF,
}

class ShaderBase(torch.nn.Module):
    def forward(self, coords_xyz, viewdirs, feat_color):
        """
        Takes color features, view directopns, and optionally coordinates at which the features were sampled,
        and produces rgb values for each point.
        :param coords_xyz:
        :param viewdirs:
        :param feat_color:
        :return:
        """
        pass


class ShaderSphericalHarmonics(ShaderBase):
    def __init__(self, sh_basis_dim, checks=False):
        super().__init__()
        self.sh_basis_dim = sh_basis_dim
        self.checks = checks

    def forward(self, coords_xyz, viewdirs, feat_rgb):
        """
        :param coords_xyz (torch.Tensor): sampled points of shape [batch x ray x 3]
        :param viewdirs (torch.Tensor): directions corresponding to inputs rays of shape [batch x 3]
        :param feat_rgb (torch.Tensor): directions corresponding to inputs rays of shape [batch x ray x rgb_feat_dim]
        :return:
        """
        # if self.checks:
        #     assert feat_rgb.dim() == 3
        #     B, R, X = feat_rgb.shape
        #     assert X == 3 * self.sh_basis_dim
        #     assert viewdirs.shape == (B, 3)
        # else:
        if len(feat_rgb.shape) == 3:
            B, R, _ = feat_rgb.shape
            rgb = feat_rgb.view(B, R, 3, self.sh_basis_dim)  # B x R x 3 x SH
            sh_mult = spherical_harmonics_bases(self.sh_basis_dim, viewdirs)  # B x SH
            sh_mult = sh_mult.view(B, 1, 1, self.sh_basis_dim)  # B x 1 x 1 x SH
            rgb = (rgb * sh_mult).sum(dim=-1)  # B x R x 3
        else:
            B, _ = feat_rgb.shape
            rgb = feat_rgb.view(B, 3, self.sh_basis_dim)  # B x R x 3 x SH
            sh_mult = spherical_harmonics_bases(self.sh_basis_dim, viewdirs)  # B x SH
            sh_mult = sh_mult.view(B, 1, self.sh_basis_dim)  # B x 1 x 1 x SH
            rgb = (rgb * sh_mult).sum(dim=-1)  # B x 3

        return rgb


class ShaderSimple(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, coords_xyz, viewdirs, features):
        return features

class ShaderMLP(torch.nn.Module):
    def __init__(self, rgb_feature_dim, posenc_viewdirs=2, posenc_feat=2, dim_latent=128, checks=False):
        super().__init__()
        self.posenc_viewdirs = posenc_viewdirs
        self.posenc_feat = posenc_feat
        self.checks = checks

        in_mlpC = (2 * posenc_viewdirs + 1) * 3 + (2 * posenc_feat + 1) * rgb_feature_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_mlpC, dim_latent),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_latent, dim_latent),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim_latent, 3),
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)
        with torch.no_grad():
            mag_scale = 10
            num_linear = 3
            scale_linear = math.pow(mag_scale, 1 / num_linear)
            for layer in self.mlp:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight *= scale_linear


    def forward(self, coords_xyz, viewdirs, feat_rgb):
        """
        :param coords_xyz (torch.Tensor): sampled points of shape [batch x ray x 3]
        :param viewdirs (torch.Tensor): directions corresponding to inputs rays of shape [batch x 3]
        :param feat_rgb (torch.Tensor): directions corresponding to inputs rays of shape [batch x ray x rgb_feat_dim]
        :return:
        """
        if self.checks:
            assert feat_rgb.dim() == 3
            B, R, X = feat_rgb.shape
            assert X == 3 * self.sh_basis_dim
            assert viewdirs.shape == (B, 3)
        else:
            B, R, _ = feat_rgb.shape

        viewdirs = viewdirs.view(-1, 1, 3).repeat(1, R, 1)
        indata = [feat_rgb, viewdirs]
        if self.posenc_feat > 0:
            indata += [positional_encoding(feat_rgb, self.posenc_feat)]
        if self.posenc_viewdirs > 0:
            indata += [positional_encoding(viewdirs, self.posenc_viewdirs)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)

        return rgb


class RadianceField(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dtype_sz_bytes = 4

        if args.shading_mode == 'spherical_harmonics':
            self.shader = ShaderSphericalHarmonics(args.sh_basis_dim, checks=args.checks)
        elif args.shading_mode == 'mlp':
            self.shader = ShaderMLP(args.rgb_feature_dim)
        elif args.shading_mode == "indetity":
            self.shader = ShaderSimple()
        else:
            raise ValueError(f'Invalid shading mode "{args.shading_mode}"')
        self.shader_num_params = sum(torch.tensor(a.shape).prod().item() for a in self.shader.parameters())


        # if args.models is None:
        #     if args.model == "QTTNF" or args.model == "TTNF":
        #         kwargs = dict(
        #             tt_rank_max=args.tt_rank_max,
        #             sample_by_contraction=args.sample_by_contraction,
        #             tt_rank_equal=args.tt_rank_equal,
        #             tt_minimal_dof=args.tt_minimal_dof,
        #             init_method=args.init_method,
        #             outliers_handling='zeros',
        #             expected_sample_batch_size=args.N_rand * (args.N_samples + args.N_importance),
        #             version_sample_qtt=args.sample_qtt_version,
        #             dtype={
        #                 'float16': torch.float16,
        #                 'float32': torch.float32,
        #                 'float64': torch.float64,
        #             }[args.dtype],
        #             checks=args.checks,
        #             verbose=True,
        #         )
        #         model=QTTNF
        #     elif args.model == "TackerNF":
        #         model = TackerNF
        #         kwargs = dict(
        #             tacker_rank=args.tacker_rank,
        #             outliers_handling='zeros',
        #         )
        #     elif args.model == "VMNF":
        #         model = VMNF
        #         kwargs = dict(
        #             vm_rank=args.vm_rank,
        #             outliers_handling='zeros',
        #         )
        #     elif args.model == "SkeletonNF":
        #         model = SkeletonNF
        #         kwargs = dict(
        #             skeleton_rank=args.skeleton_rank,
        #             outliers_handling='zeros',
        #         )
        #     elif args.model == "FullNF":
        #         model = FullNF
        #         kwargs = dict(
        #             outliers_handling='zeros',
        #         )

        #     if args.grid_type == 'fused':
        #         # opacity + 3 * (# sh or a float per channel)
        #         dim_payload = 1 + 3 * (args.sh_basis_dim if args.use_viewdirs else 1)
        #         self.vox_fused = model(args.dim_grid, dim_payload, **kwargs)
        #     elif args.grid_type == 'separate':
        #         # 3 * (# sh or a float per channel)
        #         dim_payload = 3 * (args.sh_basis_dim if args.use_viewdirs else 1)
        #         self.vox_rgb = model(
        #             args.dim_grid, dim_payload, **kwargs
        #         )
        #         self.vox_sigma = model(args.dim_grid, 1, **kwargs)  # opacity
        #     else:
        #         raise ValueError(f'Invalid voxel grid type "{args.grid_type}"')
        # else:

        models_config = json.loads(args.models)
        self.models_config = models_config
        def create_model_kwargs(model_config):
            config = deepcopy(model_config)
            del config["model"]
            return config

        if "sigma" in models_config:
            dim_payload = 3 * (args.sh_basis_dim if args.use_viewdirs else 1)
            self.vox_rgb = model_dict[models_config["rgb"]["model"]](
                args.dim_grid, dim_payload, **create_model_kwargs(models_config["rgb"])
            )
            self.vox_sigma = model_dict[models_config["sigma"]["model"]](
                args.dim_grid, 1, **create_model_kwargs(models_config["sigma"])
            )
        else:
            dim_payload = 1 + 3 * (args.sh_basis_dim if args.use_viewdirs else 1)
            self.vox_rgb = model_dict[models_config["rgb"]["model"]](
                args.dim_grid, dim_payload, **create_model_kwargs(models_config["rgb"])
            )
        if "var" in models_config:
            self.vox_var = model_dict[models_config["var"]["model"]](
                args.dim_grid, 1, **create_model_kwargs(models_config["var"])
            )
        self.bkgd_rgb = 1. if args.white_bkgd else 0.
        self.bkgd_var = torch.nn.Parameter(torch.tensor(1.))
    

    def forward(self, coords_xyz, viewdirs):
        if self.args.checks:
            assert coords_xyz.dim() == 3 and coords_xyz.shape[2] == 3

        B, R, _ = coords_xyz.shape
        coords_xyz = coords_xyz.view(B * R, 3)

        var = None
        # if self.args.models is not None:
        rgb, mask = self.vox_rgb(coords_xyz)
        if "sigma" in self.models_config:
            sigma = self.vox_sigma(coords_xyz)
        else:
            rgb, sigma = rgb[..., :-1], rgb[..., -1]  # B x R x 3 * (SH or 1), B x R
        
        rgb, sigma = rgb.view(B, R, -1), sigma.view(B, R)
        rgb = self.shader(coords_xyz, viewdirs, rgb)
        if self.args.use_rgb_sigmoid:
            rgb = torch.sigmoid(rgb)  # NR x NS x 3
        rgb = F.pad(rgb, (0, 0, 0, 1), mode="constant", value=self.bkgd_rgb)
        mask = mask.view(B, R)

        if "var" in self.models_config: 
            var = self.vox_var(coords_xyz)
            var = var.view(B, R)
            var = torch.cat([var, self.bkgd_var.expand(B, 1)], dim=1)

        # elif self.args.grid_type == 'fused':
        #     tmp = self.vox_fused(coords_xyz)
        #     rgb, sigma = tmp[..., :-1], tmp[..., -1]  # B x R x 3 * (SH or 1), B x R
        # elif self.args.grid_type == 'separate':
        #     rgb = self.vox_rgb(coords_xyz)
        #     sigma = self.vox_sigma(coords_xyz)
        # else:
        #     raise ValueError(f'Invalid grid type: "{self.args.grid_type}"')

        return rgb, sigma, var, mask
    
    def render_rays(
            self,
            ray_batch,
            N_samples,
            cur_step=None,
    ):
        N_rays, device = ray_batch.shape[0], ray_batch.device
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
        near, far = ray_batch[:, 6:7], ray_batch[:, 7:8]
        viewdirs = rays_d.clone()

        t_far = torch.linspace(0., 1.0, steps=N_samples, device=device)
        t_near = 1.0 - t_far
        z_vals = near * t_near + far * t_far
        z_vals = z_vals.expand([N_rays, N_samples])

        if self.args.perturb:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        z_vals = (z_vals * (self.args.dim_grid - 1) * 0.5).detach()
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # NR x (NS-1)

        B, R, _ = pts.shape
        coords_xyz = pts.view(B * R, 3)

        mask = torch.all(pts >= -1, dim=-1) & torch.all(pts <= 1, dim=-1)

        sigma = torch.zeros(B, R, device=pts.device, dtype=pts.dtype)
        if mask.any():
            if "sigma" in self.models_config:
                raw_sigma = self.vox_sigma(pts[mask].detach()).view(-1)
            else:
                raw_sigma = self.vox_rgb.calc_sigma(pts[mask].detach())
            
            sigma[mask] = raw_sigma
        
        # if self.args.sigma_warmup_sts and cur_step <= self.args.sigma_warmup_numsteps:
        #     sigma = sigma * (cur_step / self.args.sigma_warmup_numsteps)

        if self.args.sigma_activation == 'relu':
            sigma = F.relu(sigma.clone())  # NR x NS
            integrate_fn = integrate_new
            weights = integrate_fn(sigma, dists)  # NR x NS
        elif self.args.sigma_activation == "logloss":
            sigma = F.softplus(sigma)
            integrate_fn = integrate_new
            weights = integrate_fn(sigma, dists)  # NR x NS
        
        app_mask = mask
        if self.args.weight_threshold is not None:
            app_mask = app_mask & (weights > self.args.weight_threshold)
        
        # weights[~app_mask] = 0
        rgb = torch.zeros(B, R, 3, device=pts.device, dtype=pts.dtype)

        if app_mask.any():
            if "sigma" in self.models_config:
                raw_rgb = self.vox_rgb(pts[app_mask])
            else:
                raw_rgb = self.vox_rgb.calc_rgb(pts[app_mask])
            
            raw_rgb = self.shader(pts[app_mask], viewdirs[:,None,:].repeat(1, R, 1)[app_mask], raw_rgb)

            rgb[app_mask] = raw_rgb

        
        if self.args.use_rgb_sigmoid:
            rgb = torch.sigmoid(rgb)  # NR x NS x 3
        rgb = F.pad(rgb, (0, 0, 0, 1), mode="constant", value=self.bkgd_rgb)
            
        weights = torch.cat([weights, 1. - weights.sum(dim=1)[:,None]], dim=1)

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        return rgb_map, weights

    def get_param_groups(self):
        if self.args.optimizer == "SeparatedLBFGS":
            out = []
            if "var" in self.models_config:
                out += self.vox_var.get_param_groups()
            out += self.vox_rgb.get_param_groups()
            if "sigma" in self.models_config:
                out += self.vox_sigma.get_param_groups()
            if self.args.shading_mode == "mlp":
                out += [
                    {'tag': 'shader', 'params': self.shader.parameters(), 'lr': self.args.lrate_shader},
                ]
            return out
        out = []
        # if self.args.models is not None:
        out += [
            {'tag': 'vox', 'params': self.vox_rgb.parameters(), 'lr': self.args.lrate},
        ]
        if "sigma" in self.models_config:
            out += [
                {'tag': 'vox', 'params': self.vox_sigma.parameters(), 'lr':
                    self.args.lrate * self.args.lrate_sigma_multiplier},
            ]
        if "var" in self.models_config:
            out += [
                {'tag': 'vox', 'params': self.vox_var.parameters(), 'lr': self.args.lrate},
            ]

        # elif self.args.grid_type == 'fused':
        #     out += [
        #         {'tag': 'vox', 'params': self.vox_fused.parameters(), 'lr': self.args.lrate},
        #     ]
        # elif self.args.grid_type == 'separate':
        #     out += [
        #         {'tag': 'vox', 'params': self.vox_rgb.parameters(), 'lr': self.args.lrate},
        #         {'tag': 'vox', 'params': self.vox_sigma.parameters(), 'lr':
        #             self.args.lrate * self.args.lrate_sigma_multiplier},
        #     ]
        if self.args.shading_mode == "mlp":
            out += [
                {'tag': 'shader', 'params': self.shader.parameters(), 'lr': self.args.lrate_shader},
            ]
        return out

    @property
    def num_uncompressed_params(self):
        return sum([model.num_uncompressed_params for model in [self.vox_rgb] + \
            ([self.vox_sigma] if "sigma" in self.models_config else []) + \
            ([self.vox_var] if "var" in self.models_config else [])])

    @property
    def num_compressed_params(self):
        return sum([model.num_compressed_params for model in [self.vox_rgb] + \
            ([self.vox_sigma] if "sigma" in self.models_config else []) + \
            ([self.vox_var] if "var" in self.models_config else [])]) + \
            self.shader_num_params

    @property
    def sz_uncompressed_gb(self):
        return sum([model.sz_uncompressed_gb for model in [self.vox_rgb] + \
            ([self.vox_sigma] if "sigma" in self.models_config else []) + \
            ([self.vox_var] if "var" in self.models_config else [])]) 

    @property
    def sz_compressed_gb(self):
        return sum([model.sz_compressed_gb for model in [self.vox_rgb] + \
            ([self.vox_sigma] if "sigma" in self.models_config else []) + \
            ([self.vox_var] if "var" in self.models_config else [])]) + \
            self.shader_num_params * self.dtype_sz_bytes / (1024**3)

    @property
    def compression_factor(self):
        return self.num_uncompressed_params / self.num_compressed_params

    def get_intersect_coords(self, rays_o, rays_d):
        eps = self.args.intersect_threshold
        eps *= 2 / self.args.dim_grid
        invdirs = torch.reciprocal(rays_d)

        lb = torch.tensor([-1.] * 3, device=rays_d.device)
        rt = torch.tensor([1.] * 3, device=rays_d.device)
        t1 = (lb - rays_o) * invdirs
        t2 = (rt - rays_o) * invdirs
        near = torch.max(torch.min(t1, t2), dim=-1).values
        far = torch.min(torch.max(t1, t2), dim=-1).values

        borders = torch.linspace(-1, 1, self.args.dim_grid + 1, device=rays_o.device)[None,:].expand(3, self.args.dim_grid + 1)
        t, indices = ((borders[None,:,:] - rays_o[:,:,None]) * invdirs[:,:,None]).view(-1, 3 * (self.args.dim_grid + 1)).sort(dim=1)
        z = 0.5 * (t[:,1:] + t[:,:-1])
        dists = t[:, 1:] - t[:,:-1]
        mask = (z <= far[:,None]) & (z >= near[:,None]) & (dists >= eps) & (z >= 0)

        return z, dists, mask

