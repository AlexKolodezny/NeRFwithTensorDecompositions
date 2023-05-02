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

class KroneckerNF(BaseNF):
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
        super(KroneckerNF, self).__init__(dim_grid, dim_payload, **kwargs)

        self.tt_rank = tt_rank

        if older_size is None:
            self.older_size = int(math.sqrt(self.dim_grid))
        else:
            self.older_size = older_size

        self.younger_size = self.dim_grid // self.older_size

        self.younger_matrix = Parameter(torch.empty((1, self.tt_rank) +  (self.younger_size,) * 3))
        self.older_matrix = Parameter(torch.empty((self.older_size,) * 3 + (self.tt_rank, self.output_features)).view(-1, self.tt_rank, self.output_features))

        self.scale = scale

        self.reset_parameters()

        self.calc_params()


    def calculate_std(self) -> float:
        scale = self.scale
        return scale
        # return torch.exp(1/3 * (torch.tensor(scale).log() - 0.5 * torch.tensor(self.rank).log()))

    def reset_parameters(self) -> None:
        std = self.calculate_std()
        nn.init.normal_(self.younger_matrix, mean=0, std=std)
        nn.init.normal_(self.older_matrix, mean=0, std=std)
    
    def sample_tensor_points(self, coords_xyz):
        num_samples, _ = coords_xyz.shape
        # coo_cube, w = self.get_cubes_and_weights(coords_xyz)
        # coo_cube = coo_cube.view(8 * num_samples,3)
        # w = w.view(8 * num_samples)

        older_xyz = coords_xyz / (self.dim_grid - 1)
        q = 1 / self.older_size # TODO delete [0]
        younger_xyz = (older_xyz % q) / q * 2 - 1
        older_xyz = torch.clip(torch.floor(older_xyz * self.older_size), max=self.older_size - 1).int()

        yM = F.grid_sample(
                F.pad(self.younger_matrix, (0, 1, 0, 1, 0, 1), mode='circular'),
                younger_xyz[None,:,None,None,:].detach(),
                align_corners=True, mode='bilinear').squeeze().T

        M = batched_indexed_gemv(
                yM,
                self.older_matrix,
                (older_xyz * torch.tensor([self.older_size * self.older_size, self.older_size, 1], device=older_xyz.device)).sum(dim=1).detach()
            )

        return M

    def contract(self):
        pass