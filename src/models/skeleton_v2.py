import torch
from torch import nn

from .base_nf import BaseNF
import torch.nn.functional as F
from .tt_core import batched_indexed_gemv

class SkeletonV2NF(BaseNF):
    def __init__(
        self,
        dim_grid,
        dim_payload,
        skeleton_rank=None,
        channel_rank=None,
        scale=1.,
        **kwargs,
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        # factory_kwargs = {"device": device, "dtype": dtype}
        super(SkeletonV2NF, self).__init__(dim_grid, dim_payload, **kwargs)

        self.ranks = (skeleton_rank,) * self.dim
        self.channel_ranks = (channel_rank,) * self.dim

        self.vectors = nn.ParameterList([
            nn.Parameter(torch.empty((self.shape[i], self.ranks[i], self.channel_ranks[i])))
            for i in range(len(self.shape))
        ])
        self.matrices = nn.ParameterList([
            nn.Parameter(torch.empty(self.ranks[i:i+1] + self.shape[:i] + self.shape[i+1:]))
            for i in range(len(self.shape))
        ])

        self.channels = nn.Parameter(torch.empty(self.output_features, sum(self.channel_ranks)))

        self.scale = scale

        self.reset_parameters()

        self.calc_params()


    def calculate_std(self) -> float:
        return torch.exp(1/3 * (torch.tensor(self.scale).log() - 0.5 * (torch.tensor(self.ranks) * torch.tensor(self.channel_ranks)).sum().log()))

    def reset_parameters(self) -> None:
        std = self.calculate_std()
        for i, _ in enumerate(self.vectors):
            nn.init.normal_(self.vectors[i], mean=0, std=std)
            nn.init.normal_(self.matrices[i], mean=0, std=std)
        nn.init.normal_(self.channels, mean=0, std=std)

    
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

        results = torch.cat([
            (batched_indexed_gemv(
            # (tt_batched_indexed_gemv(
                M[:,None,:].repeat(1,2,1).view(num_samples * 2, -1),
                vector,
                coo.view(num_samples * 2)
            ).view(num_samples, 2, channel_rank) * w[:,:,None]).sum(1)
            for coo, w, M, vector, channel_rank in zip((xs, ys, zs), (wx, wy, wz), Ms, self.vectors, self.channel_ranks)
        ], dim=-1)

        return F.linear(results, self.channels)
    
    def contract(self):
        return F.linear(torch.cat([
            torch.einsum("irc,rjk->ijkc", self.vectors[0], self.matrices[0]),
            torch.einsum("jrc,rik->ijkc", self.vectors[1], self.matrices[1]),
            torch.einsum("krc,rij->ijkc", self.vectors[2], self.matrices[2])],
            dim=-1), self.channels).permute(3, 0, 1, 2)

    def get_param_groups(self):
        out = []
        out += [
            {'params': self.vectors}
        ]
        out += [
            {'params': self.matrices}
        ]
        out += [
            {'params': self.channels}
        ]
        return out
