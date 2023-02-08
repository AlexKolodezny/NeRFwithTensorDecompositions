import torch
from torch import nn

from .base_nf import BaseNF
import torch.nn.functional as F

class FullNF(BaseNF):
    def __init__(
        self,
        dim_grid,
        dim_payload,
        init_method="normal",
        constant=None,
        **kwargs,
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        # factory_kwargs = {"device": device, "dtype": dtype}
        super(FullNF, self).__init__(dim_grid, dim_payload, **kwargs)

        self.tensor = nn.Parameter(torch.empty((dim_payload,) + self.shape))
        self.init_method = init_method
        self.constant = constant

        self.reset_parameters()

        self.calc_params()


    def calculate_std(self) -> float:
        scale = 1.
        return scale

    def reset_parameters(self) -> None:
        if self.init_method == "normal":
            std = self.calculate_std()
            nn.init.normal_(self.tensor, mean=0, std=std)
        elif self.init_method == "constant":
            nn.init.constant_(self.tensor, self.constant)

    
    def sample_tensor_points(self, coords_xyz):
        num_samples, _ = coords_xyz.shape

        coords_xyz = coords_xyz / (self.dim_grid - 1) * 2 - 1
        
        result = F.grid_sample(self.tensor.unsqueeze(0), coords_xyz.view(1, num_samples, 1, 1, -1), align_corners=True).view(-1, num_samples).T
        return result
    
    def contract(self, new_tensor=False):
        if new_tensor:
            return self.new_tensor
        else:
            return self.tensor

    def get_param_groups(self):
        out = []
        out += [
            {'params': self.tensor}
        ]
        return out


def index_add_with_log_alphas(tensor_list, tensor_max_log_alphas, input_list, input_max_log_alphas, dim, indices):
    new_max_log_alphas = torch.full(tensor_max_log_alphas.shape, fill_value=-float("inf"), device=tensor_max_log_alphas.device)
    new_max_log_alphas.index_reduce_(dim, indices, input_max_log_alphas, reduce="amax", include_self=False)
    max_log_alphas = torch.max(tensor_max_log_alphas, new_max_log_alphas)
    max_log_alphas_for_input = max_log_alphas.index_select(dim, indices)
    for tensor, input in zip(tensor_list, input_list):
        tensor *= torch.exp(torch.nan_to_num(tensor_max_log_alphas - max_log_alphas))[:,None]
        tensor.index_add_(dim, indices, input * torch.exp(input_max_log_alphas - max_log_alphas_for_input)[:,None])
    return max_log_alphas


class FullNFForRGB(FullNF):
    def __init__(
        self,
        dim_grid,
        dim_payload,
        norm_var_reg=float("inf"),
        eps=0,
        log_sum_trick=True,
        **kwargs,
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        # factory_kwargs = {"device": device, "dtype": dtype}
        super(FullNFForRGB, self).__init__(dim_grid, dim_payload, **kwargs)

        self.eps = eps
        self.norm_var_reg = norm_var_reg
        self.matrix_statistic = None
        self.vector_statistic = None
        self.log_sum_trick = log_sum_trick

    
    @torch.no_grad()
    def to_tensor_coords(self, coords_xyz):
        x, y, z = torch.floor((coords_xyz + 1) / 2 * self.dim_grid).clip(0, 255).long().chunk(3, dim=1)
        x = x.squeeze()
        y = y.squeeze()
        z = z.squeeze()
        return x, y, z

    # @torch.no_grad()
    # def update_statistics(self, coords_xyz, alphas, d, target):
    #     '''
    #         coords: N_samples x 3
    #         alphas: N_samples
    #         d: N_samples x sh_dim
    #         x: N_samples x 3
    #     '''
    #     x, y, z = self.to_tensor_coords(coords_xyz)
    #     assert torch.all(alphas >= 0)
    #     self.matrix_statistic[x, y, z] += d[:,None,:] * d[:,:,None] * alphas[:,None,None] # X x Y x Z x sh_dim x sh_dim
    #     self.vector_statistic[x, y, z] += d[:,None,:] * alphas[:,None,None] * target[:,:,None] # X x Y x Z x 3 x sh_dim

    @torch.no_grad()
    def update_statistics(self, coords_xyz, log_alphas, d, target):
        '''
            coords: N_samples x 3
            alphas: N_samples
            d: N_samples x sh_dim
            x: N_samples x 3
        '''
        assert torch.all(log_alphas <= 0)
        # print(torch.min(log_alphas))
        x, y, z = self.to_tensor_coords(coords_xyz)
        coords = z + self.shape[-1] * (y + self.shape[-2]  * x)
        if self.log_sum_trick:
            self.max_log_alphas = index_add_with_log_alphas(
                [
                    self.matrix_statistic.view(self.shape[0] * self.shape[1] * self.shape[2], -1),
                    self.vector_statistic.view(self.shape[0] * self.shape[1] * self.shape[2], -1)
                ],
                self.max_log_alphas.view(self.shape[0] * self.shape[1] * self.shape[2]),
                [(d[:,None,:] * d[:,:,None]).view(d.shape[0], -1), (d[:,None,:] * target[:,:,None]).view(d.shape[0], -1)],
                log_alphas, 0, coords).view(self.shape)
        else:
            self.matrix_statistic[x, y, z] += d[:,None,:] * d[:,:,None] * torch.exp(log_alphas)[:,None,None] # X x Y x Z x sh_dim x sh_dim
            self.vector_statistic[x, y, z] += d[:,None,:] * torch.exp(log_alphas)[:,None,None] * target[:,:,None] # X x Y x Z x 3 x sh_dim
    
    @torch.no_grad()
    def calc_new_tensor(self, var):
        if self.log_sum_trick:
            self.matrix_statistic += self.eps * torch.eye(self.dim_payload // 3, device=self.matrix_statistic.device)[None,None,None,:,:]
            cho_factors = torch.linalg.cholesky(
                self.matrix_statistic * torch.exp(self.max_log_alphas)[...,None,None] + \
                    (var[:,:,:,None,None] / self.norm_var_reg) * torch.eye(self.dim_payload // 3, device=self.matrix_statistic.device)[None,None,None,:,:]
                ).unsqueeze(3) # X x Y x Z x 1 x sh_dim x sh_sim
            self.new_tensor = torch.cholesky_solve((self.vector_statistic * torch.exp(self.max_log_alphas[...,None,None])).unsqueeze(5), cho_factors)\
                .view(self.shape + (self.dim_payload,)).permute(3, 0, 1, 2)
        else:
            cho_factors = torch.linalg.cholesky(
                self.matrix_statistic + \
                    (var[:,:,:,None,None] / self.norm_var_reg + self.eps) * torch.eye(self.dim_payload // 3, device=self.matrix_statistic.device)[None,None,None,:,:]
                ).unsqueeze(3) # X x Y x Z x 1 x sh_dim x sh_sim
            self.new_tensor = torch.cholesky_solve(self.vector_statistic.unsqueeze(5), cho_factors)\
                .view(self.shape + (self.dim_payload,)).permute(3, 0, 1, 2)
    
    @torch.no_grad()
    def update_old_tensor(self):
        self.tensor.data = self.new_tensor
    
    def update_tensor(self, var):
        self.tensor.data = torch.tensor(0.)
        self.calc_new_tensor(var)
        self.tensor.data = self.new_tensor
        del self.new_tensor
        if self.log_sum_trick:
            del self.max_log_alphas
        del self.matrix_statistic
        del self.vector_statistic
        torch.cuda.empty_cache()
    
    @torch.no_grad()
    def calc_rgb(self, coords_xyz, d, new_tensor=False):
        x, y, z = self.to_tensor_coords(coords_xyz)
        tensor = self.contract(new_tensor)
        return torch.sum(tensor[:,x,y,z].T.view(x.shape[0], 3, -1) * d[:,None,:], dim=2)
    
    @torch.no_grad()
    def zeros_statistics(self):
        if self.log_sum_trick:
            self.max_log_alphas = torch.full(self.shape, fill_value=-float("inf"), device=self.tensor.device)
        self.matrix_statistic = torch.zeros(self.shape + (self.dim_payload // 3, self.dim_payload // 3), device = self.tensor.device)
        self.vector_statistic = torch.zeros(self.shape + (3, self.dim_payload // 3), device=self.tensor.device)

    @torch.no_grad()
    def log_reg(self, new_tensor):
        tensor = self.contract(new_tensor)
        return -0.5 / self.norm_var_reg * torch.sum(tensor**2, dim=0)


class FullNFForSigma(FullNF):
    def __init__(
        self,
        dim_grid,
        dim_payload,
        beta_a_reg=0,
        beta_b_reg=0,
        log_sum_trick=True,
        **kwargs,
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        # factory_kwargs = {"device": device, "dtype": dtype}
        assert dim_payload == 1
        super(FullNFForSigma, self).__init__(dim_grid, dim_payload, **kwargs)

        self.beta_a_reg = beta_a_reg
        self.beta_b_reg = beta_b_reg
        self.pass_statistics = None
        self.not_pass_statistics = None
        self.log_sum_trick = log_sum_trick

    @torch.no_grad()
    def zeros_statistics(self):
        if self.log_sum_trick:
            self.max_log_pass_statistics = torch.full(self.shape, fill_value=-float("inf"), device=self.tensor.device)
            self.max_log_not_pass_statistics = torch.full(self.shape, fill_value=-float("inf"), device=self.tensor.device)
        self.pass_statistics = torch.zeros(self.shape + (1,), device=self.tensor.device)
        self.not_pass_statistics = torch.zeros(self.shape + (1,), device=self.tensor.device)

    @torch.no_grad()
    def to_tensor_coords(self, coords_xyz):
        x, y, z = torch.floor((coords_xyz + 1) / 2 * self.dim_grid).clip(0, 255).long().chunk(3, dim=1)
        x = x.squeeze()
        y = y.squeeze()
        z = z.squeeze()
        return x, y, z

    @torch.no_grad()
    def update_statistics(self, coords_xyz, log_p, log_np):
        x, y, z = self.to_tensor_coords(coords_xyz)
        if self.log_sum_trick:
            coords = z + self.shape[-1] * (y + self.shape[-2]  * x)
            self.max_log_pass_statistics = index_add_with_log_alphas(
                [
                    self.pass_statistics.view(self.shape[0] * self.shape[1] * self.shape[2], -1),
                ],
                self.max_log_pass_statistics.view(self.shape[0] * self.shape[1] * self.shape[2]),
                [torch.ones_like(log_p)[:,None]],
                log_p, 0, coords).view(self.shape)
            self.max_log_not_pass_statistics = index_add_with_log_alphas(
                [
                    self.not_pass_statistics.view(self.shape[0] * self.shape[1] * self.shape[2], -1),
                ],
                self.max_log_not_pass_statistics.view(self.shape[0] * self.shape[1] * self.shape[2]),
                [torch.ones_like(log_np)[:,None]],
                log_np, 0, coords).view(self.shape)
        else:
            self.pass_statistics[x,y,z] += torch.exp(log_p)[...,None]
            self.not_pass_statistics[x,y,z] += torch.exp(log_np)[...,None]
    
    def update_old_tensor(self):
        self.tensor.data = self.new_tensor
    
    def update_tensor(self):
        self.tensor.data = torch.tensor(0.)
        self.calc_new_tensor()
        self.tensor.data = self.new_tensor
        del self.new_tensor
        del self.not_pass_statistics
        del self.pass_statistics
        if self.log_sum_trick:
            del self.max_log_pass_statistics
            del self.max_log_not_pass_statistics
        torch.cuda.empty_cache()
    
    @torch.no_grad()
    def calc_new_tensor(self):
        if self.log_sum_trick:
            self.new_tensor = (
                torch.log(self.not_pass_statistics * torch.exp(self.max_log_not_pass_statistics)[...,None] + self.beta_a_reg) - \
                (torch.log(self.pass_statistics * torch.exp(self.max_log_pass_statistics)[...,None] + self.beta_b_reg))
            ).view((1,) + self.shape)
        else:
            self.new_tensor = (
                torch.log(self.not_pass_statistics + self.beta_a_reg) - \
                (torch.log(self.pass_statistics + self.beta_b_reg))
            ).view((1,) + self.shape)
    
    def calc_sigma(self, coords_xyz, new_tensor=False):
        x, y, z = self.to_tensor_coords(coords_xyz)
        tensor = self.contract(new_tensor)
        return tensor.squeeze()[x, y, z]
    
    def log_reg(self, new_tensor=False):
        tensor = self.contract(new_tensor)
        return - self.beta_a_reg * F.softplus(-tensor) - self.beta_b_reg * F.softplus(tensor)
    
    def log_bernulli(self, new_tensor=False):
        tensor = self.contract(new_tensor)
        if self.log_sum_trick:
            return - self.not_pass_statistics.squeeze() * torch.exp(self.max_log_not_pass_statistics) * F.softplus(-tensor)\
                - self.pass_statistics.squeeze() * torch.exp(self.max_log_pass_statistics) * F.softplus(tensor)
        else:
            return - self.not_pass_statistics.squeeze() * F.softplus(-tensor)\
                - self.pass_statistics.squeeze() * F.softplus(tensor)

    

class FullNFForVar(FullNF):
    def __init__(
        self,
        dim_grid,
        dim_payload,
        gamma_a_reg=0,
        gamma_b_reg=0,
        log_sum_trick=True,
        **kwargs,
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        # factory_kwargs = {"device": device, "dtype": dtype}
        assert dim_payload == 1
        super(FullNFForVar, self).__init__(dim_grid, dim_payload, **kwargs)

        self.error_statistics = None
        self.sum_alphas = None

        self.gamma_a_reg = gamma_a_reg
        self.gamma_b_reg = gamma_b_reg

        self.var_bkgd = nn.Parameter(torch.ones((1,1)))
        self.log_sum_trick = log_sum_trick

    @torch.no_grad()
    def zeros_statistics(self):
        self.bkgd_error_statistics = torch.full((1,1), 0., device=self.tensor.device)
        self.bkgd_sum_alphas = torch.full((1,1), 0., device=self.tensor.device)
        self.error_statistics = torch.zeros(self.shape + (1,), device=self.tensor.device)
        self.sum_alphas = torch.zeros(self.shape + (1,), device=self.tensor.device)
        if self.log_sum_trick:
            self.bkgd_max_log_alpha = torch.full((1,), -float("inf"), device=self.tensor.device)
            self.max_log_alphas = torch.full(self.shape, fill_value=-float("inf"), device=self.tensor.device)

    @torch.no_grad()
    def to_tensor_coords(self, coords_xyz):
        x, y, z = torch.floor((coords_xyz + 1) / 2 * self.dim_grid).clip(0, 255).long().chunk(3, dim=1)
        x = x.squeeze()
        y = y.squeeze()
        z = z.squeeze()
        return x, y, z

    @torch.no_grad()
    def update_statistics(self, coords_xyz, log_alphas, square_errors):
        assert torch.all(log_alphas <= 0)
        x, y, z = self.to_tensor_coords(coords_xyz)
        if self.log_sum_trick:
            coords = z + self.shape[-1] * (y + self.shape[-2]  * x)
            self.max_log_alphas = index_add_with_log_alphas(
                [
                    self.error_statistics.view(self.shape[0] * self.shape[1] * self.shape[2], -1),
                    self.sum_alphas.view(self.shape[0] * self.shape[1] * self.shape[2], -1)
                ],
                self.max_log_alphas.view(self.shape[0] * self.shape[1] * self.shape[2]),
                [square_errors[:,None], torch.ones_like(square_errors)[:,None]],
                log_alphas, 0, coords).view(self.shape)
        else:
            self.error_statistics[x,y,z] += torch.exp(log_alphas)[...,None] * square_errors[...,None]
            self.sum_alphas[x,y,z] += torch.exp(log_alphas)[...,None]
        return
    
    @torch.no_grad()
    def update_bkgd_statistics(self, log_alphas, square_errors):
        assert torch.all(log_alphas <= 0)
        if self.log_sum_trick:
            self.bkgd_max_log_alpha = index_add_with_log_alphas(
                [
                    self.bkgd_error_statistics.view(1, 1),
                    self.bkgd_sum_alphas.view(1, 1),
                ],
                self.bkgd_max_log_alpha,
                [square_errors[:,None], torch.ones_like(square_errors)[:,None]],
                log_alphas, 0, torch.zeros_like(square_errors).long()).view(1)
        else:
            self.bkgd_error_statistics += torch.sum(torch.exp(log_alphas) * square_errors)[...,None]
            self.bkgd_sum_alphas += torch.sum(torch.exp(log_alphas))[...,None]
    

    @torch.no_grad()
    def calc_new_tensor(self):
        if self.log_sum_trick:
            self.new_tensor = ((1.5 * self.error_statistics * torch.exp(self.max_log_alphas)[...,None] + self.gamma_b_reg) /\
                (1.5 * self.sum_alphas * torch.exp(self.max_log_alphas)[...,None] + self.gamma_a_reg)).view((1,) + self.shape)
            self.new_var_bkgd = ((1.5 * self.bkgd_error_statistics * torch.exp(self.bkgd_max_log_alpha) + self.gamma_b_reg) /\
                (1.5 * self.bkgd_sum_alphas * torch.exp(self.bkgd_max_log_alpha) + self.gamma_a_reg))
        else:
            self.new_tensor = ((1.5 * self.error_statistics + self.gamma_b_reg) /\
                (1.5 * self.sum_alphas + self.gamma_a_reg)).view((1,) + self.shape)
            self.new_var_bkgd = ((1.5 * self.bkgd_error_statistics + self.gamma_b_reg) /\
                (1.5 * self.bkgd_sum_alphas + self.gamma_a_reg))
    
    def update_old_tensor(self):
        self.tensor.data = self.new_tensor
        self.var_bkgd.data = self.new_var_bkgd
    
    def update_tensor(self):
        self.tensor.data = torch.tensor(0.)
        self.var_bkgd.data = torch.tensor(0.)
        self.calc_new_tensor()
        self.tensor.data = self.new_tensor
        self.var_bkgd.data = self.new_var_bkgd
        del self.new_tensor
        del self.new_var_bkgd
        del self.error_statistics
        del self.sum_alphas
        del self.bkgd_error_statistics
        del self.bkgd_sum_alphas
        if self.log_sum_trick:
            del self.max_log_alphas
            del self.bkgd_max_log_alpha
        torch.cuda.empty_cache()
    
    def calc_var(self, coords_xyz, new_tensor=False):
        x, y, z = self.to_tensor_coords(coords_xyz)
        tensor = self.contract(new_tensor)
        return tensor.squeeze()[x, y, z]

    def calc_var_bkgd(self, new_tensor=False):
        if new_tensor:
            return self.new_var_bkgd
        else:
            return self.var_bkgd
    
    def log_gamma_reg(self, new_tensor=False):
        tensor = self.contract(new_tensor).squeeze()
        return -self.gamma_a_reg * torch.log(tensor) - self.gamma_b_reg / tensor
    
    def log_gamma_reg_bkgd(self, new_tensor=False):
        bkgd = self.calc_var_bkgd(new_tensor)
        return -self.gamma_a_reg * torch.log(bkgd) - self.gamma_b_reg / bkgd
    
    def log_normal(self, new_tensor=False):
        var = self.contract(new_tensor).squeeze()
        if self.log_sum_trick:
            return - 1.5 * (self.sum_alphas.squeeze() * torch.exp(self.max_log_alphas) * torch.log(var)) \
                - 1.5 / (var) * (self.error_statistics.squeeze() * torch.exp(self.max_log_alphas))
        else:
            return - 1.5 * self.sum_alphas.squeeze() * torch.log(var) \
                - 1.5 / (var) * self.error_statistics.squeeze()
    
    def log_normal_bkgd(self, new_tensor=False):
        bkgd = self.calc_var_bkgd(new_tensor)
        if self.log_sum_trick:
            return - 1.5 * (self.bkgd_sum_alphas * torch.exp(self.bkgd_max_log_alpha)) * torch.log(bkgd) \
                - 1.5 / (bkgd) * (self.bkgd_error_statistics * torch.exp(self.bkgd_max_log_alpha))
        else:
            return - 1.5 * (self.bkgd_sum_alphas * torch.log(bkgd) \
                - 1.5 / (bkgd) * (self.bkgd_error_statistics))

    