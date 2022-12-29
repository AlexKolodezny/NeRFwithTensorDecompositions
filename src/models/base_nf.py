import torch
from torch import nn
import torch.nn.functional as F
import typing
from torch import Tensor


def batched_indexed_gemv_with_stacking(bv, mm, m_indices, return_unordered=False, checks=False, stacking=False):
    if checks:
        if not (torch.is_tensor(bv) and torch.is_tensor(mm) and torch.is_tensor(m_indices)):
            raise ValueError('Operand is not a tensor')
        if not (bv.dtype == mm.dtype and m_indices.dtype is torch.uint8):
            raise ValueError(f'Incompatible dtypes: {bv.dtype=} {mm.dtype=} {m_indices.dtype=}')
        if bv.dim() != 2 or mm.dim() != 3 or m_indices.dim() != 1 or bv.shape[0] != m_indices.shape[0] \
                or bv.shape[1] != mm.shape[1]:
            raise ValueError(f'Invalid operand shapes: {bv.shape=} {mm.shape=} {m_indices.shape=}')
    m_indices_uniq_vals, m_indices_uniq_cnts = m_indices.unique(sorted=True, return_counts=True)
    # assert len(m_indices_uniq_vals) == mm.shape[0]
    m_indices_order_fwd = m_indices.argsort()
    m_indices_splited = m_indices_order_fwd.tensor_split(m_indices_uniq_cnts.cumsum(0).cpu()[:-1])
    m_indices_uniq_vals = m_indices_uniq_vals.long()

    if checks:
        if m_indices_uniq_vals.max() >= mm.shape[0]:
            raise ValueError('Incompatible index and matrices')

    mx = m_indices_uniq_cnts.max()
    num_samples = bv.shape[0]
    batch_count = len(m_indices_splited)

    m_indices_stacked = torch.cat([
        F.pad(v_in, (0, mx - v_in.shape[0]), "constant", num_samples)
        for v_in in m_indices_splited
    ], dim=0)
    bwd = m_indices_stacked.argsort()[:num_samples]

    bv_stacked = torch.cat([bv, torch.zeros(1, bv.shape[1], device=bv.device)])\
        .index_select(0, m_indices_stacked.detach())\
        .view(batch_count, mx, -1)
    bv_out = torch.bmm(bv_stacked.view(batch_count, mx, -1), mm.index_select(0, m_indices_uniq_vals))\
        .view(batch_count * mx, -1)\
        .index_select(0, bwd.detach())
    return bv_out


class BaseNF(nn.Module):
    def __init__(
        self,
        dim_grid,
        dim_payload,
        outliers_handling="zeros",
            
    ) -> None:
        # assert len(dim_grid) == len(ranks)
        self.dim = 3
        self.dtype_sz_bytes = 4
        self.outliers_handling = outliers_handling
        assert self.outliers_handling == "zeros", f"Unsupported outliers_handling {self.outliers_handling}"

        # factory_kwargs = {"device": device, "dtype": dtype}
        super(BaseNF, self).__init__()

        self.dim_grid = dim_grid
        self.shape = (dim_grid,) * self.dim
        self.output_features = dim_payload

    
    def calc_params(self):
        self.num_uncompressed_params = torch.prod(torch.tensor(self.shape)).item()
        self.num_compressed_params = sum([torch.prod(torch.tensor(p.shape)) for p in self.parameters()]).item()
        self.sz_uncompressed_gb = self.num_uncompressed_params * self.dtype_sz_bytes / (1024 ** 3)
        self.sz_compressed_gb = self.num_compressed_params * self.dtype_sz_bytes / (1024 ** 3)
        self.compression_factor = self.num_uncompressed_params / self.num_compressed_params
    
    def calc_simple_std(self, target_sigma, ranks, num_tensors):
        return (1 / num_tensors * (torch.tensor(target_sigma).log() - \
            0.5 * torch.tensor(ranks).log().sum())).exp()

    def calculate_std(self) -> float:
        pass

    def reset_parameters(self) -> None:
        pass
    

    def get_cubes_and_weights(self, coords_xyz: Tensor) -> Tensor:
        num_samples = coords_xyz.shape[0]
        offs = torch.tensor([0, 1], device=coords_xyz.device)

        coo_left_bottom_near = torch.floor(coords_xyz)
        coo_left_bottom_near = coo_left_bottom_near.int()

        coo_right_up_far = coo_left_bottom_near + 1

        coo_right_up_far.clamp_max_(self.dim_grid - 1)
        coo_left_bottom_near.clamp_max_(self.dim_grid - 1)

        xs, ys, zs = tuple(
            torch.stack([coo_left_bottom_near[:,i], coo_right_up_far[:,i]], dim=1)
            for i in range(3)
        )

        wa = coords_xyz - coo_left_bottom_near
        wb = coo_right_up_far  - coords_xyz

        wx, wy, wz = tuple(
            torch.stack([wa[:,i], wb[:,i]], dim=1)
            for i in range(3)
        )

        return xs, ys, zs, wx, wy, wz
    
    
    def coords_to_tensor(self, coords_xyz: Tensor) -> Tensor:
        return (coords_xyz + 1) * (0.5 * (self.dim_grid - 1))
    

    def sample_with_outlier_handling(self, coords_xyz: Tensor) -> Tensor:
        """
        :param coords_xyz (torch.Tensor): sampled float points of shape [batch_rays x 3] in cube [0, dim_grid-1]^3
        :return:
        """

        batch_size, _ = coords_xyz.shape
        if self.outliers_handling == 'zeros':
            mask_valid = torch.all(coords_xyz >= 0, dim=1) & torch.all(coords_xyz <= self.dim_grid - 1, dim=1)
            coords_xyz = coords_xyz[mask_valid]
            if coords_xyz.shape[0] == 0:
                return torch.zeros(batch_size, self.output_features, dtype=coords_xyz.dtype, device=coords_xyz.device)
            mask_need_remap = coords_xyz.shape[0] < batch_size
        

        result = self.sample_tensor_points(coords_xyz)

        
        if self.outliers_handling == 'zeros' and mask_need_remap:
            out_sparse = torch.zeros(batch_size, self.output_features, dtype=coords_xyz.dtype, device=coords_xyz.device)
            out_sparse[mask_valid] = result
            return out_sparse

        return result
    
    def sample_tensor_points(self, coords_xyz):
        pass

    
    def forward(self, coords_xyz: Tensor) -> Tensor:
        """
        :param coords_xyz (torch.Tensor): sampled float points of shape [batch_rays x 3] in cube [0, dim_grid-1]^3
        :return:
        """
        return self.sample_with_outlier_handling(self.coords_to_tensor(coords_xyz))
        