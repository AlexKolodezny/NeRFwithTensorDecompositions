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

from .base_nf import BaseNF

class TTCPNF(BaseNF):
    def __init__(
        self,
        dim_grid,
        dim_payload,
        rank=None,
        scale=0.1,
        **kwargs,
            
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        # factory_kwargs = {"device": device, "dtype": dtype}
        super(TTCPNF, self).__init__(dim_grid, dim_payload, **kwargs)

        self.rank = rank

        self.older_size = int(math.sqrt(self.dim_grid))
        self.younger_size = self.dim_grid // self.older_size

        self.older = nn.Parameter(torch.empty((1,rank) + (self.older_size,) * 3))
        self.younger = nn.Parameter(torch.empty((1,rank) + (self.younger_size,) * 3))

        self.B = nn.Linear(self.rank, self.output_features, bias=False)

        self.scale = scale

        self.reset_parameters()

        self.calc_params()


    def calculate_std(self) -> float:
        scale = self.scale
        return scale
        # return torch.exp(1/3 * (torch.tensor(scale).log() - 0.5 * torch.tensor(self.rank).log()))

    def reset_parameters(self) -> None:
        std = self.calculate_std()
        nn.init.normal_(self.older, mean=0, std=std)
        nn.init.normal_(self.younger, mean=0, std=std)
        nn.init.normal_(self.B.weight, mean=0, std=std)

    
    def sample_tensor_points(self, coords_xyz):
        num_samples, _ = coords_xyz.shape
        # coo_cube, w = self.get_cubes_and_weights(coords_xyz)
        # coo_cube = coo_cube.view(8 * num_samples,3)
        # w = w.view(8 * num_samples)

        older_xyz = coords_xyz / (self.dim_grid - 1)
        q = 1 / self.older_size
        younger_xyz = (older_xyz % q) / q * 2 - 1
        older_xyz = older_xyz * 2 - 1

        A = F.grid_sample(
            self.older,
            older_xyz[None,:,None,None,:].detach(),
            align_corners=False, mode='nearest').view(-1, num_samples)

        B = F.grid_sample(
            F.pad(self.younger, (0, 1, 0, 1, 0, 1), mode='circular'),
            younger_xyz[None,:,None,None,:].detach(),
            align_corners=True, mode='bilinear').view(-1, num_samples)

        return self.B((A * B).T)

    def contract(self):
        return torch.einsum(
            "aijk,ca->cijk",
            (self.older[:,:,:,None,:,None,:,None] * self.younger[:,:,None,:,None,:,None,:]).view(self.rank, self.dim_grid, self.dim_grid, self.dim_grid),
            self.B.weight)