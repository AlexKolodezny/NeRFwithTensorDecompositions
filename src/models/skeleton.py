import torch
from torch import nn

from .base_nf import BaseNF
import torch.nn.functional as F
from .tt_core import batched_indexed_gemv

class SkeletonNF(BaseNF):
    def __init__(
        self,
        dim_grid,
        dim_payload,
        skeleton_rank=None,
        outliers_handling="zeros",
            
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        # factory_kwargs = {"device": device, "dtype": dtype}
        super(SkeletonNF, self).__init__(dim_grid, dim_payload, outliers_handling=outliers_handling)

        self.ranks = (skeleton_rank,) * self.dim

        self.vectors = nn.ParameterList([
            nn.Parameter(torch.empty((self.shape[i], self.ranks[i], self.output_features)))
            for i in range(len(self.shape))
        ])
        self.matrices = nn.ParameterList([
            nn.Parameter(torch.empty(self.ranks[i:i+1] + self.shape[:i] + self.shape[i+1:]))
            for i in range(len(self.shape))
        ])

        self.reset_parameters()

        self.calc_params()


    def calculate_std(self) -> float:
        scale = 1.
        return torch.exp(1/2 * (torch.tensor(scale).log() - 0.5 * torch.tensor(self.ranks).sum().log()))

    def reset_parameters(self) -> None:
        std = self.calculate_std()
        for i, _ in enumerate(self.vectors):
            nn.init.normal_(self.vectors[i], mean=0, std=std)
            nn.init.normal_(self.matrices[i], mean=0, std=std)

    
    def sample_tensor_points(self, coords_xyz):
        num_samples, _ = coords_xyz.shape
        xs, ys, zs, wx, wy, wz = self.get_cubes_and_weights(coords_xyz)

        coords_xyz = coords_xyz / (self.dim_grid - 1) * 2 - 1
        
        Ms = [
            F.grid_sample(
                matrix[None,:,:,:],
                coords_xyz[None,:,None,[y,z]].detach(),
                align_corners=True).squeeze().T
            for (x, y, z), matrix in zip([(0, 1, 2), (1, 0, 2), (2, 0, 1)], self.matrices)
        ]

        results = [
            (batched_indexed_gemv(
            # (tt_batched_indexed_gemv(
                M[:,None,:].repeat(1,2,1).view(num_samples * 2, -1),
                vector,
                coo.view(num_samples * 2)
            ).view(num_samples, 2, self.output_features) * w[:,:,None]).sum(1)
            for coo, w, M, vector in zip((xs, ys, zs), (wx, wy, wz), Ms, self.vectors)
        ]

        return sum(results)
