from .tt_core import *
from .base_nf import BaseNF
import numpy as np

def coord_tensor_to_coord_tt(coords_xyz, dim_modes, chunk=False, checks=False):
    if checks:
        if not torch.is_tensor(coords_xyz) or coords_xyz.dim() != 2 or coords_xyz.shape[1] != 3:
            raise ValueError('Coordinates is not an Nx3 tensor')
        if not coords_xyz.dtype in (torch.int, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('Coordinates are not integer')
        if torch.any(coords_xyz < 0) or torch.any(coords_xyz > np.prod(dim_modes) - 1):
            raise ValueError('Coordinates out of bounds')
    dim_modes = torch.tensor(dim_modes, device=coords_xyz.device)
    factors = F.pad(torch.cumprod(dim_modes, 0), pad=(1,0), mode='constant', value=1) # modes
    bits_xyz = torch.div(coords_xyz.unsqueeze(-1), factors[:-1], rounding_mode='floor') % dim_modes # N * 3 * modes
    octets = dim_modes[None,:]**torch.arange(0, 3, device=coords_xyz.device)[:,None] # 3 * modes
    core_indices = (bits_xyz * octets[None,:,:]).sum(dim=1).long()
    if chunk:
        core_indices = core_indices.chunk(len(dim_modes), dim=1)  # [core_0_ind, ..., core_last_ind]
        core_indices = [c.view(-1) for c in core_indices]
    return core_indices

class TTNF(BaseNF):
    def __init__(
            self,
            dim_grid,
            dim_payload,
            dim_modes=None,
            tt_rank_max=None,
            tt_rank_equal=False,
            tt_minimal_dof=True,
            init_method='normal',
            version_sample_qtt=3,
            dtype=torch.float32,
            checks=False,
            verbose=False,
            **kwargs,
    ):
        super().__init__(dim_grid, dim_payload, **kwargs)
        if init_method not in ('zeros', 'eye', 'normal'):
            raise ValueError('init_method can be either zeros, eye, or normal')
        if type(dim_modes) not in (tuple, list):
            raise ValueError(f'Invalid TT modes {dim_modes}')
        if version_sample_qtt == 3 and not tt_minimal_dof:
            raise ValueError('version_sample_qtt=3 requires tt_minimal_dof=True')
        self.tt_minimal_dof = tt_minimal_dof
        self.init_method = init_method
        self.version_sample_qtt = version_sample_qtt
        self.dtype = dtype
        self.checks = checks
        self.verbose = verbose

        self.dim_modes = dim_modes
        self.tt_modes = list(map(lambda x: x**3, self.dim_modes)) + [dim_payload]
        self.tt_rank_max = tt_rank_max
        self.num_cores = len(self.tt_modes)
        self.tt_ranks = get_tt_ranks(self.tt_modes, max_rank=tt_rank_max, tt_rank_equal=tt_rank_equal)

        self.tt_core_shapes = [
            (self.tt_ranks[i], self.tt_modes[i], self.tt_ranks[i + 1])
            for i in range(self.num_cores)
        ]

        if tt_minimal_dof:
            self.tt_core_isparam = [s[0] * s[1] != s[2] and s[0] != s[1] * s[2] for s in self.tt_core_shapes]
            if not any(self.tt_core_isparam):
                # full rank case, need to appoint one (largest) core as a parameter
                core_sizes = [s[0] * s[1] * s[2] for s in self.tt_core_shapes]
                largest_core_idx = core_sizes.index(max(core_sizes))
                self.tt_core_isparam[largest_core_idx] = True
        else:
            self.tt_core_isparam = [True] * self.num_cores

        for i in range(self.num_cores):
            if self.tt_core_isparam[i]:
                self.register_parameter(
                    self._get_core_name_by_id(i),
                    torch.nn.Parameter(torch.zeros(*self.tt_core_shapes[i], dtype=dtype))
                )
            else:
                core_shape = self.tt_core_shapes[i]
                if core_shape[0] == core_shape[1] * core_shape[2]:
                    eye_size = core_shape[0]
                else:
                    eye_size = core_shape[2]
                buf_init = torch.eye(eye_size, dtype=dtype).reshape(core_shape)
                self.register_buffer(self._get_core_name_by_id(i), buf_init)

        if init_method == 'normal':
            num_buffers_on_the_left = self.tt_core_isparam.index(True)
            num_buffers_on_the_right = list(reversed(self.tt_core_isparam)).index(True)
            ranks_between_two_param_cores = self.tt_ranks[1 + num_buffers_on_the_left: -1 - num_buffers_on_the_right]
            d = sum([int(a) for a in self.tt_core_isparam])
            self.sigma_cores = (-torch.tensor(ranks_between_two_param_cores).double().log().sum() / (2. * d)).exp().item()
            for i, c in enumerate(self.get_cores()):
                if not self.tt_core_isparam[i]:
                    continue
                with torch.no_grad():
                    c.copy_((torch.randn_like(c) * self.sigma_cores).to(dtype))
        elif init_method == 'eye':
            for i, c in enumerate(self.get_cores()):
                if not self.tt_core_isparam[i]:
                    continue
                with torch.no_grad():
                    c.copy_(torch.eye(
                        self.tt_ranks[i], self.tt_ranks[i + 1], dtype=dtype
                    ).unsqueeze(1).repeat(1, self.tt_modes[i], 1))

        if self.tt_core_isparam is None or all(self.tt_core_isparam):
            version_sample_qtt = 2
        self.fn_sample_intcoord = {
            2: partial(sample_intcoord_tt_v2, last_core_is_payload=True),
            3: partial(sample_intcoord_tt_v3, last_core_is_payload=True, tt_core_isparam=self.tt_core_isparam),
        }[version_sample_qtt]

        self.dtype_sz_bytes = {
            torch.float16: 2,
            torch.float32: 4,
            torch.float64: 8,
        }[self.dtype]
        self.num_uncompressed_params = torch.prod(torch.tensor(self.tt_modes)).item()
        self.num_compressed_params = sum([torch.prod(torch.tensor(p.shape)) for p in self.parameters()]).item()
        self.sz_uncompressed_gb = self.num_uncompressed_params * self.dtype_sz_bytes / (1024 ** 3)
        self.sz_compressed_gb = self.num_compressed_params * self.dtype_sz_bytes / (1024 ** 3)
        self.compression_factor = self.num_uncompressed_params / self.num_compressed_params

    def sample_tensor_points(self, coords):
        B, _ = coords.shape
        xs, ys, zs, wx, wy, wz = self.get_cubes_and_weights(coords)
        coords = torch.stack([
            xs[:,:,None,None].repeat(1, 1, 2, 2),
            ys[:,None,:,None].repeat(1, 2, 1, 2),
            zs[:,None,None,:].repeat(1, 2, 2, 1),
        ], dim=4).view(B * 8, 3)
        coords = coord_tensor_to_coord_tt(coords, self.dim_modes, chunk=True, checks=self.checks)
        bv = self.fn_sample_intcoord(self.get_cores(), coords, checks=self.checks).view(B, 2, 2, 2,-1)
        return (bv * wx[:,:,None,None,None] * wy[:,None,:,None,None] * wz[:,None,None,:,None]).view(B, 8, -1).sum(dim=1)

    @staticmethod
    def _get_core_name_by_id(i):
        return f'core{i:02d}'

    def _get_core(self, i):
        return getattr(self, self._get_core_name_by_id(i))

    def get_cores(self):
        return [self._get_core(i) for i in range(self.num_cores)]

    def init_with_decomposition(self, A):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                category=UserWarning,
            )
            try:
                import tntorch
            except ImportError:
                raise ImportError('tntorch is required only in this one function')
            A_hat = tntorch.Tensor(A, ranks_tt=self.tt_ranks[1:-1])
        with torch.no_grad():
            for i in range(self.num_cores):
                if self.tt_core_shapes[i] != A_hat.cores[i].shape:
                    raise ValueError(f'Incompatible core shapes: {self.tt_core_shapes=} {shapes(A_hat.cores)=}')
                self._get_core(i).copy_(A_hat.cores[i])
            if self.tt_minimal_dof:
                self.reduce_parameterization(self.get_cores())

    @staticmethod
    @torch.no_grad()
    def reduce_parameterization(cores):
        if not is_tt(cores):
            raise ValueError('Input is not a Tensor Train')
        for i, c in enumerate(cores):
            if c.shape[0] * c.shape[1] != c.shape[2] or i == len(cores) - 1:
                break
            r = c.shape[2]
            i_neigh = i + 1
            c_neigh = cores[i_neigh]
            s_neigh = c_neigh.shape
            c_neigh_new = c.view(-1, r).mm(c_neigh.view(r, -1)).view(s_neigh)
            c_neigh.copy_(c_neigh_new)
            c.copy_(torch.eye(r, device=c.device, dtype=c.dtype).view(c.shape))
        for i, c in enumerate(reversed(cores)):
            if c.shape[0] != c.shape[1] * c.shape[2] or i == len(cores) - 1:
                break
            r = c.shape[0]
            i_neigh = len(cores) - 2 - i
            c_neigh = cores[i_neigh]
            s_neigh = c_neigh.shape
            c_neigh_new = c_neigh.view(-1, r).mm(c.view(r, -1)).view(s_neigh)
            c_neigh.copy_(c_neigh_new)
            c.copy_(torch.eye(r, device=c.device, dtype=c.dtype).view(c.shape))

    def contract(self):
        """
        Computes the entire uncompressed voxel grid.
        Caution: may cause out-of-memory.
        :return: torch.Tensor of shape (dim_grid, dim_grid, dim_grid, dim_payload).
        """
        out = convert_qtt_to_tensor(
            self.get_cores(),
            qtt_reshape_plan=None,
            fn_contract=self.fn_contract_grid,
            checks=self.checks,
        )
        return out

    def extra_repr(self) -> str:
        core_shapes_status = ', '.join([
            f"{c} ({'param' if p else 'buffer'})"
            for c, p in zip(self.tt_core_shapes, self.tt_core_isparam)
        ])
        return \
            f'number of uncompressed parameters: {self.num_uncompressed_params}\n' + \
            f'number of compressed parameters: {self.num_compressed_params}\n' + \
            f'size uncompressed: {self.sz_uncompressed_gb:.3f} Gb\n' \
            f'size compressed: {self.sz_compressed_gb:.3f} Gb\n' \
            f'compression factor: {self.compression_factor:.3f}x\n' + \
            f'core shapes: {core_shapes_status}\n' + \
            f'dtype: {self.dtype}'
