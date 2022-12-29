import torch


def log_integrate(log_passed, log_not_passed):
    log_passed = F.pad(log_passed[:, :-1], (1, 0))  # NR x NS (add column zeros left, remove column right)
    log_alpha = log_not_passed  # NR x NS
    cum_log_att_exclusive = torch.cumsum(log_passed, dim=1)  # NR x NS
    return torch.cat([log_alpha + cum_log_att_exclusive, log_passed.sum(dim=1)[:,None]], dim=1)


class Renderer(nn.Module):
    def __init__(
            self,
            radiance_field,
            perturb=0,
            white_bkgd=False,
            use_rgb_sigmoid=None,
            sigma_activation='logloss',
        ):
        super().__init__()
        
        self.radiance_field = radiance_field
        self.perturb = perturb
        self.white_bkgd = white_bkgd

    
    def render_rays(
            self,
            ray_batch,
            target_batch,
            model,
            N_samples,
            use_viewdirs=False,
            dim_grid=None,
            white_bkgd=False,
            sigma_warmup_sts=False,
            sigma_warmup_numsteps=None,
            cur_step=None,
            checks=True,
            use_lrf=None,
            sigma_activation='relu',
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

        def sample_rays(inputs, viewdirs):
            """
            Prepares inputs and applies network 'fn'.
            :param inputs (torch.Tensor): sampled points of shape [batch x ray x 3]
            :param viewdirs (torch.Tensor): directions corresponding to inputs rays of shape [batch x 3]
            """
            rgb, sigma = model(inputs, viewdirs)
            if sigma_warmup_sts and cur_step <= sigma_warmup_numsteps:
                sigma *= cur_step / sigma_warmup_numsteps
            return rgb, sigma

        N_rays, device = ray_batch.shape[0], ray_batch.device
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
        near, far = ray_batch[:, 6:7], ray_batch[:, 7:8]
        viewdirs = None
        if use_viewdirs:
            viewdirs = rays_d.clone()

        t_far = torch.linspace(0., 1., steps=N_samples, device=device)
        t_near = 1. - t_far
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

        raw_rgb, raw_sigma = sample_rays(pts, viewdirs)

        if use_lrf:
            assert N_importance == 0
            out = march_rays_lrf(
                raw_rgb,
                raw_sigma,
                z_vals,
                target_batch,
                raw_noise_std,
                white_bkgd,
                ret_weights=N_importance > 0,
                use_rgb_sigmoid=use_rgb_sigmoid,
                sigma_activation=sigma_activation)
        else:
            out = march_rays(raw_rgb, raw_sigma, z_vals, raw_noise_std, white_bkgd, ret_weights=N_importance > 0, use_rgb_sigmoid=use_rgb_sigmoid)

        return out


        def calc_loss_and_render(
                self,
                raw_rgb,
                rgb_sigma,
                raw_sigma,
                dists,
                targets,
                ret_weights=False,
        ):
            assert self.use_rgb_sigmoid is not None
            assert self.sigma_activation == 'logloss'

            NR, NS = raw_sigma.shape

            print(raw_sigma.sum(), raw_rgb.sum())

            sigma = raw_sigma * dists
            log_passed = -torch.logaddexp(raw_sigma, torch.zeros_like(raw_sigma))
            log_not_passed = -torch.logaddexp(-raw_sigma, torch.zeros_like(raw_sigma))

            # print(log_passed.sum(), log_not_passed.sum())

            if self.use_rgb_sigmoid:
                rgb = torch.sigmoid(raw_rgb)  # NR x NS x 3
            else:
                rgb = raw_rgb

            log_weights = log_integrate(log_passed, log_not_passed)
            weights = log_weights.exp()

            norm_sigma = torch.ones(1, device=raw_sigma.device)

            bkgd_color = 1 if white_bkgd else 0
            rgb = F.pad(rgb, (0, 0, 0, 1), mode="constant", value=bkgd_color)
            # print(log_weights.sum())

            log_p = -1.5 * torch.log(norm_sigma) - 1/(2 * norm_sigma) * torch.sum((rgb - targets[:,None,:])**2, dim=2) + log_weights
            q = torch.softmax(log_p, dim=1).detach()
            log_q = F.log_softmax(log_p, dim=1).detach()
            # print((log_weights != float("inf")).sum())
            # print(q.sum())
            # print(log_p.sum())

            # q = log_p.exp()
            # q /= q.sum(dim=1)

            elbo = (log_p * q).sum() - (q * log_q).sum()
            print("ELBO: ", elbo.item())

            rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
            # rgb_map = rgb_map + bkgd_color * (1. - acc_map[..., None])

            out = {
                'rgb_map': rgb_map,
                'losses': torch.sum(-torch.where(q == 0, torch.zeros_like(q), log_p * q), dim=1)
            }
            # print(out['losses'].sum())
            if ret_weights:
                out['weights'] = weights

            return out

