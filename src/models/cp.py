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

class CPNF(BaseNF):
    def __init__(
        self,
        dim_grid,
        dim_payload,
        cp_rank=None,
        scale=1,
        **kwargs,
            
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        # factory_kwargs = {"device": device, "dtype": dtype}
        super(CPNF, self).__init__(dim_grid, dim_payload, **kwargs)

        self.ranks = (cp_rank,) * self.dim

        self.vectors = nn.ParameterList([
            Parameter(torch.empty((1,self.ranks[i], self.shape[i],1)))
            for i in range(len(self.shape))
        ])
        self.B = nn.Linear(sum(self.ranks), self.output_features, bias=False)

        self.scale = scale

        self.reset_parameters()

        self.calc_params()


    def calculate_std(self) -> float:
        scale = 1.
        return 0.1
        # return torch.exp(1/3 * (torch.tensor(scale).log() - 0.5 * torch.tensor(self.ranks).sum().log()))

    def reset_parameters(self) -> None:
        std = self.calculate_std()
        print("Std", std)
        for i, _ in enumerate(self.vectors):
            nn.init.normal_(self.vectors[i], mean=0, std=std)
        nn.init.normal_(self.B.weight, mean=0, std=std)

    
    def sample_tensor_points(self, coords_xyz):
        num_samples, _ = coords_xyz.shape
        # coo_cube, w = self.get_cubes_and_weights(coords_xyz)
        # coo_cube = coo_cube.view(8 * num_samples,3)
        # w = w.view(8 * num_samples)

        coords_xyz = coords_xyz / (self.dim_grid - 1) * 2 - 1

        V = [
            F.grid_sample(
                vector,
                torch.stack([torch.zeros_like(coords_xyz[None,:,None,x]), coords_xyz[None,:,None,x], ], dim=3).detach(),
                align_corners=True).view(-1, num_samples)
            for (x, y, z), vector in zip([(0, 1, 2), (1, 0, 2), (2, 0, 1)], self.vectors)
        ]

        if self.output_features != 1:
            return self.B((V[0] * V[1] * V[2]).T)
        else:
            return (V[0] * V[1] * V[2]).T.sum(dim=1, keepdim=True)
