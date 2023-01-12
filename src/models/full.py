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

        self.tensor = nn.Parameter(torch.empty((1, dim_payload) + self.shape))
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
        xs, ys, zs, wx, wy, wz = self.get_cubes_and_weights(coords_xyz)

        coords_xyz = coords_xyz / (self.dim_grid - 1) * 2 - 1
        
        result = F.grid_sample(self.tensor, coords_xyz[None,:,None,None,:].detach(), align_corners=True).squeeze(4).squeeze(3).squeeze(0).T
        return result
