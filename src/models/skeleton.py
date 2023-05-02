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
        symmetric=True,
        full_interpolation=True,
        scale=1.,
        **kwargs,
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        # factory_kwargs = {"device": device, "dtype": dtype}
        super(SkeletonNF, self).__init__(dim_grid, dim_payload, **kwargs)

        self.full_interpolation = full_interpolation

        if symmetric:
            self.ranks = (skeleton_rank,) * self.dim
        else:
            self.ranks = (skeleton_rank,)

        if self.output_features != 1:
            self.vectors = nn.ParameterList([
                nn.Parameter(torch.empty((self.shape[i], self.ranks[i], self.output_features)))
                for i in range(len(self.ranks))
            ])
        else:
            self.vectors = nn.ParameterList([
                nn.Parameter(torch.empty((1,self.ranks[i], self.shape[i],1)))
                for i in range(len(self.ranks))
            ])
        self.matrices = nn.ParameterList([
            nn.Parameter(torch.empty(self.ranks[i:i+1] + self.shape[:i] + self.shape[i+1:]))
            for i in range(len(self.ranks))
        ])

        self.scale = scale

        self.reset_parameters()

        self.calc_params()


    def calculate_std(self) -> float:
        return torch.exp(1/2 * (torch.tensor(self.scale).log() - 0.5 * torch.tensor(self.ranks).sum().log()))

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
                align_corners=True).view(-1, num_samples).T
            for (x, y, z), matrix in zip([(0, 1, 2), (1, 0, 2), (2, 0, 1)], self.matrices)
        ]
        if self.output_features != 1:
            if self.full_interpolation:
                results = [
                    (batched_indexed_gemv(
                        M[:,None,:].repeat(1,2,1).view(num_samples * 2, -1),
                        vector,
                        coo.view(num_samples * 2)
                    ).view(num_samples, 2, self.output_features) * w[:,:,None]).sum(1)
                    for coo, w, M, vector in zip((xs, ys, zs), (wx, wy, wz), Ms, self.vectors)
                ]
            else:
                results = [
                    (batched_indexed_gemv(
                        M.view(num_samples, -1),
                        vector,
                        coo[:,0]
                    ).view(num_samples, self.output_features))
                    for coo, w, M, vector in zip((xs, ys, zs), (wx, wy, wz), Ms, self.vectors)
                ]

            return sum(results)
        else:
            M = torch.cat(Ms, dim=1)
            V = torch.cat([
                F.grid_sample(
                    vector,
                    torch.stack([torch.zeros_like(coords_xyz[None,:,None,x]), coords_xyz[None,:,None,x], ], dim=3).detach(),
                    align_corners=True).view(-1, num_samples)
                for (x, y, z), vector in zip([(0, 1, 2), (1, 0, 2), (2, 0, 1)], self.vectors)
            ], dim=0).T

            return (M * V).sum(dim=1, keepdim=True)

    
    def calc_sigma(self, coords_xyz):
        num_samples, _ = coords_xyz.shape
        M = torch.cat([
            F.grid_sample(
                matrix[None,:,:,:],
                coords_xyz[None,:,None,[y,z]].detach(),
                align_corners=True).view(-1, num_samples)
            for (x, y, z), matrix in zip([(0, 1, 2), (1, 0, 2), (2, 0, 1)], self.matrices)
        ], dim=0)
        V = torch.cat([
            F.grid_sample(
                (vector[...,-1].T)[None,:,:,None],
                torch.stack([torch.zeros_like(coords_xyz[None,:,None,x]), coords_xyz[None,:,None,x], ], dim=3).detach(),
                align_corners=True).view(-1, num_samples)
            for (x, y, z), vector in zip([(0, 1, 2), (1, 0, 2), (2, 0, 1)], self.vectors)
        ], dim=0)
        return (V * M).sum(dim=0)
    
    def calc_rgb(self, coords_xyz):
        num_samples, _ = coords_xyz.shape
        xs, ys, zs, wx, wy, wz = self.get_cubes_and_weights(self.coords_to_tensor(coords_xyz))

        Ms = [
            F.grid_sample(
                matrix[None,:,:,:],
                coords_xyz[None,:,None,[y,z]].detach(),
                align_corners=True).view(-1, num_samples).T
            for (x, y, z), matrix in zip([(0, 1, 2), (1, 0, 2), (2, 0, 1)], self.matrices)
        ]

        results = [
            (batched_indexed_gemv(
                M[:,None,:].repeat(1,2,1).view(num_samples * 2, -1),
                vector[...,:-1],
                coo.view(num_samples * 2)
            ).view(num_samples, 2, self.output_features - 1) * w[:,:,None]).sum(1)
            for coo, w, M, vector in zip((xs, ys, zs), (wx, wy, wz), Ms, self.vectors)
        ]

        return sum(results)


    
    def contract(self):
        if len(self.ranks) > 1:
            return \
                torch.einsum("irc,rjk->cijk", self.vectors[0], self.matrices[0]) + \
                torch.einsum("jrc,rik->cijk", self.vectors[1], self.matrices[1]) + \
                torch.einsum("krc,rij->cijk", self.vectors[2], self.matrices[2])
        else:
            return torch.einsum("irc,rjk->cijk", self.vectors[0], self.matrices[0])


    def get_param_groups(self):
        out = []
        out += [
            {'params': self.vectors}
        ]
        out += [
            {'params': self.matrices}
        ]
        return out
