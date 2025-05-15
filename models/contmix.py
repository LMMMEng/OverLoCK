'''
This is a plug-and-play implementation of ContMix block in the paper:
https://arxiv.org/abs/2502.20087
'''
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
from timm.models.layers import DropPath, to_2tuple
from torch.utils.checkpoint import checkpoint

try:
    from natten.functional import na2d_av
    has_natten = True
except:
    has_natten = False
    warnings.warn("The efficiency may be reduced since 'natten' is not installed."
                  " It is recommended to install natten for better performance.")


def get_conv2d(in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               dilation,
               groups,
               bias,
               attempt_use_lk_impl=True):

    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)

    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        x = x * self.proj(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):

        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])

        return x


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()



class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))


    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x



class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 19:
            self.kernel_sizes = [5, 7, 9, 9, 3, 3, 3]
            self.dilates = [1, 1, 1, 2, 4, 5, 7]
        elif kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 7, 5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 7, 5, 3, 3]
            self.dilates = [1, 1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class ResDWConv(nn.Conv2d):
    '''
    Depthwise conv with residual connection
    '''
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)

    def forward(self, x):
        x = x + super().forward(x)
        return x


class ContMixBlock(nn.Module):
    '''
    A plug-and-play implementation of ContMix module with FFN layer
    Paper: https://arxiv.org/abs/2502.20087
    '''
    def __init__(self,
                 dim=64,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=2,
                 mlp_ratio=4,
                 res_scale=False,
                 ls_init_value=None,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False,
                 **kwargs):

        super().__init__()
        '''
        Args:
        kernel_size: kernel size of the main ContMix branch, default is 7
        smk_size: kernel size of the secondary ContMix branch, default is 5
        num_heads: number of dynamic kernel heads, default is 2
        mlp_ratio: ratio of mlp hidden dim to embedding dim, default is 4
        res_scale: whether to use residual layer scale, default is False
        ls_init_value: layer scale init value, default is None
        drop_path: drop path rate, default is 0
        norm_layer: normalization layer, default is LayerNorm2d
        use_gemm: whether to use iGEMM implementation for large kernel conv, default is False
        deploy: whether to use deploy mode, default is False
        use_checkpoint: whether to use grad checkpointing, default is False
        **kwargs: other arguments
        '''
        mlp_dim = int(dim*mlp_ratio)
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.use_gemm = use_gemm
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.use_checkpoint = use_checkpoint

        self.dwconv1 = ResDWConv(dim, kernel_size=3)
        self.norm1 = norm_layer(dim)

        self.weight_query = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim//2),
        )
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim//2),
        )
        self.weight_value = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.weight_proj = nn.Conv2d(49, kernel_size**2 + smk_size**2, kernel_size=1)
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
        )
        self.se_layer = SEModule(dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

        self.proj = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        self.dwconv2 = ResDWConv(dim, kernel_size=3)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
        )

        self.ls1 = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.get_rpb()

    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.trunc_normal_(self.rpb1, std=0.02)
        nn.init.trunc_normal_(self.rpb2, std=0.02)

    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)

    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        """
        RPB implementation directly borrowed from https://tinyurl.com/mrbub4t3
        """
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size//2] = height - (kernel_size-1)
        num_repeat_w[kernel_size//2] = width - (kernel_size-1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*kernel_size-1)) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size**2))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size**2))
        return attn + rpb

    def reparm(self):
        for m in self.modules():
            if isinstance(m, DilatedReparamBlock):
                m.merge_dilated_branches()

    def _forward_inner(self, x):
        input_resolution = x.shape[2:]
        B, C, H, W = x.shape

        x = self.dwconv1(x)
        identity = x
        x = self.norm1(x)
        gate = self.gate(x)
        lepe = self.lepe(x)

        is_pad = False
        if min(H, W) < self.kernel_size:
            is_pad = True
            if H < W:
                size = (self.kernel_size, int(self.kernel_size / H * W))
            else:
                size = (int(self.kernel_size / W * H), self.kernel_size)
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            H, W = size

        query = self.weight_query(x) * self.scale
        key = self.weight_key(x)
        value = self.weight_value(x)

        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
        weight = self.weight_proj(weight)
        weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)

        attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size**2, self.kernel_size**2], dim=-1)
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        value = rearrange(value, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)

        if has_natten:
            x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)
            x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)
        else:
            pad1 = self.smk_size // 2
            pad2 = self.kernel_size // 2
            H_o1 = H - 2 * pad1
            W_o1 = W - 2 * pad1
            H_o2 = H - 2 * pad2
            W_o2 = W - 2 * pad2

            v1 = rearrange(value[0], 'b g h w c -> b (g c) h w')
            v2 = rearrange(value[1], 'b g h w c -> b (g c) h w')

            v1 = F.unfold(v1, kernel_size=self.smk_size).reshape(B, -1, H_o1, W_o1)
            v2 = F.unfold(v2, kernel_size=self.kernel_size).reshape(B, -1, H_o2, W_o2)

            v1 = F.pad(v1, (pad1, pad1, pad1, pad1), mode='replicate')
            v2 = F.pad(v2, (pad2, pad2, pad2, pad2), mode='replicate')

            v1 = rearrange(v1, 'b (g c k) h w -> b g c h w k', g=self.num_heads, k=self.smk_size**2, h=H, w=W)
            v2 = rearrange(v2, 'b (g c k) h w -> b g c h w k', g=self.num_heads, k=self.kernel_size**2, h=H, w=W)

            x1 = einsum(attn1, v1, 'b g h w k, b g c h w k -> b g h w c')
            x2 = einsum(attn2, v2, 'b g h w k, b g c h w k -> b g h w c')

        x = torch.cat([x1, x2], dim=1)
        x = rearrange(x, 'b g h w c -> b (g c) h w', h=H, w=W)

        if is_pad:
            x = F.adaptive_avg_pool2d(x, input_resolution)

        x = self.fusion_proj(x)

        x = x + lepe
        x = self.se_layer(x)

        x = gate * x
        x = self.proj(x)

        if self.res_scale:
            x = self.ls1(identity) + self.drop_path(x)
        else:
            x = identity + self.drop_path(self.ls1(x))

        x = self.dwconv2(x)

        if self.res_scale:
            x = self.ls2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        return x

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self._forward_inner, x, use_reentrant=False)
        else:
            x = self._forward_inner(x)
        return x


if __name__ == '__main__':

    from timm.utils import random_seed
    random_seed(6)

    x = torch.randn(1, 64, 32, 32).cuda()
    model = ContMixBlock(dim=64,
                         num_heads=2,
                         kernel_size=13,
                         smk_size=5,
                         mlp_ratio=4,
                         res_scale=True,
                         ls_init_value=1,
                         drop_path=0,
                         norm_layer=LayerNorm2d,
                         use_gemm=True,
                         deploy=False,
                         use_checkpoint=False)
    print(model)
    model.cuda()
    model.eval()
    y = model(x)
    print(y.shape)

    # Reparametrize model, more details can be found at:
    # https://github.com/AILab-CVC/UniRepLKNet/tree/main
    model.reparm()
    z = model(x)

    # Showing difference between original and reparametrized model
    print((z - y).abs().sum() / y.abs().sum())