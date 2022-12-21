from typing import Tuple
import torch
from torch import Tensor
from torch import nn
from typing import Tuple
from functools import reduce
import operator
from torch.nn import init
import math
import opt_einsum
from torch import nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.init import _no_grad_normal_

from .base_nf import BaseNF


class TackerNF(BaseNF):
    def __init__(
        self,
        dim_grid,
        dim_payload,
        tacker_rank=None,
        outliers_handling="zeros",
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        super(TackerNF, self).__init__(dim_grid, dim_payload, outliers_handling=outliers_handling)

        self.ranks = (tacker_rank,) * self.dim

        self.factors = nn.ParameterList([
            Parameter(torch.empty((self.ranks[i],self.shape[i])))
            for i in range(len(self.shape))
        ])
        self.core = Parameter(torch.empty(self.ranks + (self.output_features,)))

        self.batch_size = None
        self.compiled_expr = None

        # self.output_factor = Parameter(torch.empty((self.output_rank, self.output_features), **factory_kwargs))
        self.reset_parameters()

        self.calc_params()

    def calculate_std(self) -> float:
        return self.calc_simple_std(target_sigma=1., ranks=self.ranks, num_tensors=len(self.factors) + 1)

    def reset_parameters(self) -> None:
        std = self.calculate_std()
        print("Calced std", std)
        for i, _ in enumerate(self.factors):
            torch.nn.init.normal_(self.factors[i], 0, std)
        torch.nn.init.normal_(self.core, 0, std)
    
    def sample_tensor_points(self, coords_xyz: Tensor) -> Tensor:

        coords_xyz = coords_xyz / (self.dim_grid - 1) * 2 - 1

        Fs = [
            F.grid_sample(
                factor[None,:,:,None],
                torch.stack([coords_xyz[None,:,None,i], torch.zeros_like(coords_xyz[None,:,None,i])], dim=3).detach(),
                align_corners=True).squeeze()
            for i, factor in enumerate(self.factors)
        ]

        # print("Real std", self.core.std(), self.core.mean())

        # if self.batch_size is None:
        #     print("Create expression")
        #     self.batch_size = coords_xyz.shape[0]
        #     self.compiled_expr = opt_einsum.contract_expression(
        #         "ia,ja,ka,ijkb->ab",
        #         *[factor.shape for factor in Fs],
        #         self.core.shape,
        #         optimize='optimal',
        #     )

        return torch.einsum(
            "ia,ja,ka,ijkb->ab",
            *Fs,
            self.core,
        )
    