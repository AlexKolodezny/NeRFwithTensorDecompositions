from typing import Tuple
import torch
from torch import Tensor
from torch import nn
from typing import Tuple
from functools import reduce
import operator
from torch.nn import init
import math
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.init import _no_grad_normal_

from .tt_core import batched_indexed_gemv

from .base_nf import BaseNF

class VTTMNF(BaseNF):
    def __init__(
        self,
        dim_grid,
        dim_payload,
        cp_rank=None,
        tt_rank=None,
        older_size=None,
        scale=0.1,
        **kwargs,
            
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        # factory_kwargs = {"device": device, "dtype": dtype}
        super(VTTMNF, self).__init__(dim_grid, dim_payload, **kwargs)

        self.cp_rank = (cp_rank,) * 3
        self.tt_rank = (tt_rank,) * 3

        if older_size is None:
            self.older_size = (int(math.sqrt(self.dim_grid)),) * 3
        else:
            self.older_size = (older_size,) * 3

        self.younger_size = tuple(self.dim_grid // older_size for older_size in self.older_size)

        self.vectors = nn.ParameterList([
            Parameter(torch.empty((1,self.cp_rank[i], self.shape[i],1)))
            for i in range(len(self.shape))
        ])
        self.younger_matrices = nn.ParameterList([
            Parameter(torch.empty((1,) + self.tt_rank[i:i+1] + self.younger_size[:i] + self.younger_size[i+1:]))
            for i in range(len(self.shape))
        ])
        self.older_matrices = nn.ParameterList([
            Parameter(torch.empty(self.older_size[:i] + self.older_size[i+1:] + self.tt_rank[i:i+1] + self.cp_rank[i:i+1]).view(-1, self.tt_rank[i], self.cp_rank[i]))
            for i in range(len(self.shape))
        ])
        self.B = nn.Linear(sum(self.cp_rank), self.output_features, bias=False)

        self.scale = scale

        self.reset_parameters()

        self.calc_params()


    def calculate_std(self) -> float:
        scale = self.scale
        return scale
        # return torch.exp(1/3 * (torch.tensor(scale).log() - 0.5 * torch.tensor(self.rank).log()))

    def reset_parameters(self) -> None:
        std = self.calculate_std()
        for i, _ in enumerate(self.vectors):
            nn.init.normal_(self.vectors[i], mean=0, std=std)
            nn.init.normal_(self.younger_matrices[i], mean=0, std=std)
            nn.init.normal_(self.older_matrices[i], mean=0, std=std)
        nn.init.normal_(self.B.weight, mean=0, std=std)
    
    def sample_tensor_points(self, coords_xyz):
        num_samples, _ = coords_xyz.shape
        # coo_cube, w = self.get_cubes_and_weights(coords_xyz)
        # coo_cube = coo_cube.view(8 * num_samples,3)
        # w = w.view(8 * num_samples)

        older_xyz = coords_xyz / (self.dim_grid - 1)
        q = 1 / self.older_size[0] # TODO delete [0]
        younger_xyz = (older_xyz % q) / q * 2 - 1
        older_xyz = torch.clip(torch.floor(older_xyz * self.older_size[0]), max=self.older_size[0] - 1).int()

        yM = [
            F.grid_sample(
                F.pad(matrix, (0, 1, 0, 1), mode='circular'),
                younger_xyz[None,:,None,[y,z]].detach(),
                align_corners=True, mode='bilinear').squeeze().T
            for (x, y, z), matrix in zip([(0, 1, 2), (1, 0, 2), (2, 0, 1)], self.younger_matrices)
        ]

        M = torch.cat([
            batched_indexed_gemv(
                M,
                oM,
                older_xyz[:,y] * o_size + older_xyz[:,z],
            )
            for (x,y,z), M, oM, o_size in zip([(0, 1, 2), (1, 0, 2), (2, 0, 1)], yM, self.older_matrices, self.older_size)
        ], dim=1)

        coords_xyz = coords_xyz / (self.dim_grid - 1) * 2 - 1

        V = torch.cat([
            F.grid_sample(
                vector,
                torch.stack([torch.zeros_like(coords_xyz[None,:,None,x]), coords_xyz[None,:,None,x], ], dim=3).detach(),
                align_corners=True).view(-1, num_samples)
            for (x, y, z), vector in zip([(0, 1, 2), (1, 0, 2), (2, 0, 1)], self.vectors)
        ], dim=0).T

        return self.B(M * V)

    def contract(self):
        return torch.einsum(
            "aijk,ca->cijk",
            (self.older[:,:,:,None,:,None,:,None] * self.younger[:,:,None,:,None,:,None,:]).view(self.rank, self.dim_grid, self.dim_grid, self.dim_grid),
            self.B.weight)