import torch
from torch import nn

from .base_nf import BaseNF
import torch.nn.functional as F

class FullNF(BaseNF):
    def __init__(
        self,
        dim_grid,
        dim_payload,
        outliers_handling="zeros",
            
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        # factory_kwargs = {"device": device, "dtype": dtype}
        super(FullNF, self).__init__(dim_grid, dim_payload, outliers_handling=outliers_handling)

        self.tensor = nn.Parameter(torch.empty((1, self.dim_payload) + self.shape))

        self.reset_parameters()

        self.calc_params()


    def calculate_std(self) -> float:
        scale = 1.
        return scale

    def reset_parameters(self) -> None:
        std = self.calculate_std()
        nn.init.normal_(self.tensor, mean=0, std=std)

    
    def sample_tensor_points(self, coords_xyz):
        num_samples, _ = coords_xyz.shape
        xs, ys, zs, wx, wy, wz = self.get_cubes_and_weights(coords_xyz)

        coords_xyz = coords_xyz / (self.dim_grid - 1) * 2 - 1
        
        return F.grid_sample(tensor, coords_xyz[None,:,None,None,:].detach(), align_corners=True).squeeze(4).squeeze(3).squeeze(0)
