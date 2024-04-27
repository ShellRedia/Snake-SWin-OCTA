import torch
from torch import nn, cat, Tensor
import torch.nn.functional as F
from torch.nn.functional import dropout
import os
from collections import defaultdict
import einops
from typing import Optional, Callable, List, Any

from torchvision.ops.misc import MLP, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import WeightsEnum, Weights
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param

from torchinfo import summary


def _patch_merging_pad(x):
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    return x

torch.fx.wrap("_patch_merging_pad")


class PatchMerging(nn.Module):
    def __init__(self, dim: int=3, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        _log_api_usage_once(self)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor): # [..., H, W, C] -> [..., H/2, W/2, 2*C]
        x = x.permute(0, 2, 3, 1)
        x = _patch_merging_pad(x)
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        x = self.reduction(self.norm(x))
        x = x.permute(0, 3, 1, 2)
        return x

def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
):
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H: shift_size[0] = 0
    if window_size[1] >= pad_W: shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (C // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention")

class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).view(-1)  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor):
        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]  # type: ignore[index]
        relative_position_bias = relative_position_bias.view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
        )
    
class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 3,
        num_heads: int = 3,
        window_size: List[int] = [7, 7],
        shift_size: List[int] = [3, 3],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()
        _log_api_usage_once(self)

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
        device: str | torch.device = "cuda",
    ):
        """
        A Dynamic Snake Convolution Implementation

        Based on:

            TODO

        Args:
            in_ch: number of input channels. Defaults to 1.
            out_ch: number of output channels. Defaults to 1.
            kernel_size: the size of kernel. Defaults to 9.
            extend_scope: the range to expand. Defaults to 1 for this method.
            morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
            if_offset: whether deformation is required,  if it is False, it is the standard convolution kernel. Defaults to True.

        """

        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        # self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, input: torch.Tensor):
        # Predict offset map between [-1, 1]
        offset = self.offset_conv(input)
        # offset = self.bn(offset)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        # Run deformative conv
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        output = self.dsc_conv_y(deformed_feature) if self.morph else self.dsc_conv_x(deformed_feature)

        # Groupnorm & ReLU
        output = self.gn(output)
        output = self.relu(output)

        return output


def get_coordinate_map_2D(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
    device: str | torch.device = "cuda",
):
    """Computing 2D coordinate map of DSCNet based on: TODO

    Args:
        offset: offset predict by network with shape [B, 2*K, W, H]. Here K refers to kernel size.
        morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
        extend_scope: the range to expand. Defaults to 1 for this method.
        device: location of data. Defaults to 'cuda'.

    Return:
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
    """

    if morph not in (0, 1): raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        """
        Initialize the kernel and flatten the kernel
            y: only need 0
            x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        Initialize the kernel and flatten the kernel
            y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
            x: only need 0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")

        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """From coordinate map interpolate feature of DSCNet based on: TODO

    Args:
        input_feature: feature that to be interpolated with shape [B, C, H, W]
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
        interpolate_mode: the arg 'mode' of nn.functional.grid_sample, can be 'bilinear' or 'bicubic' . Defaults to 'bilinear'.

    Return:
        interpolated_feature: interpolated feature with shape [B, C, K_H * H, K_W * W]
    """

    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Note here grid with shape [B, H, W, 2]
    # Where [:, :, :, 2] refers to [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    """Map the value of coordinate_map from origin=[min, max] to target=[a,b] for DSCNet based on: TODO

    Args:
        coordinate_map: the coordinate map to be scaled
        origin: original value range of coordinate map, e.g. [coordinate_map.min(), coordinate_map.max()]
        target: target value range of coordinate map,Defaults to [-1, 1]

    Return:
        coordinate_map_scaled: the coordinate map after scaling
    """
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled
    
class MultiView_DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        device_id: str | torch.device = "cuda",
    ):
        super().__init__()

        device = torch.device(device_id if torch.cuda.is_available() else "cpu")
        self.dsconv_x = DSConv(in_channels, out_channels, kernel_size, extend_scope, 1, True, device_id).to(device)
        self.dsconv_y = DSConv(in_channels, out_channels, kernel_size, extend_scope, 0, True, device_id).to(device)
        self.conv = Conv(in_channels, out_channels)
        self.conv_fusion = Conv(out_channels * 3, out_channels)

    def forward(self, x):
        conv_x = self.conv(x)
        dsconvx_x = self.dsconv_x(x)
        dsconvy_x = self.dsconv_y(x)
        x = self.conv_fusion(torch.cat([conv_x, dsconvx_x, dsconvy_x], dim=1))
        return x

class SwinLayer(nn.Module):
    def __init__(self, channels=36, is_shift = 1):
        super().__init__()

        self.swin_layer = SwinTransformerBlock(channels, channels // 12, [7, 7], [3*is_shift, 3*is_shift])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.swin_layer(x)
        x = x.permute(0, 3, 1, 2)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class DSCNet(nn.Module):
    def __init__(
        self,
        img_ch=3,
        output_ch=1,
        kernel_size=5,
        extend_scope=3,
        layer_depth=5,
        rate=72,
        dim=1,
        device_id="0"
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dim = dim 
        self.layer_depth = layer_depth

        device_id = "cuda:{}".format(device_id)

        basic_feature = [2**x for x in range(layer_depth)] 
        basic_feature += basic_feature[:-1][::-1] # [1, 2, 4, 8, 4, 2, 1]

        in_channels = [img_ch] + [x*rate for x in basic_feature[:layer_depth-1]] + [3*x*rate for x in basic_feature[-layer_depth+1:]]
        out_channels = [x*rate for x in basic_feature]

        init_DSConvFusion = lambda in_ch, out_ch : MultiView_DSConv(in_ch, out_ch, kernel_size, extend_scope, device_id)

        self.down = nn.MaxPool2d(2)
        
        self.dsconvs = nn.Sequential(*[init_DSConvFusion(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, out_channels)])

        self.out_conv = nn.Conv2d(rate, output_ch, 1)
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        layer_depth = self.layer_depth

        edc_features = []
        for i in range(layer_depth):
            xi = self.dsconvs[i](x)
            if i < layer_depth - 1:
                edc_features.append(xi)
                x = self.down(xi)
            else: x = xi
        # decoder:
        x = self.up(x)
        # print([x.shape for x in edc_features])
        for i in range(1, layer_depth-1)[::-1]:
            x = torch.cat([x, edc_features[i]], dim=1)
            x = self.dsconvs[2 * (layer_depth-1) - i](x)
            x = self.up(x)
        
        x = torch.cat([x, edc_features[0]], dim=1)
        x = self.dsconvs[2 * (layer_depth-1)](x)
        out = self.out_conv(x)

        return self.sigmoid(out)

class SwinSnake_Alter(nn.Module):
    def __init__(
        self,
        img_ch=3,
        output_ch=1,
        kernel_size=5,
        extend_scope=3,
        layer_depth=5,
        rate=72,
        dim=1,
        repeat_n=1,
        down_layer="MaxPooling",
        device_id="0"
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dim = dim 
        self.layer_depth = layer_depth

        device_id = "cuda:{}".format(device_id)

        basic_feature = [2**x for x in range(layer_depth)] 
        basic_feature += basic_feature[:-1][::-1] # [1, 2, 4, 8, 4, 2, 1]

        if down_layer == "MaxPooling":
            in_channels = [img_ch] + [x*rate for x in basic_feature[:layer_depth-1]] + [3*x*rate for x in basic_feature[-layer_depth+1:]]
            out_channels = [x*rate for x in basic_feature]
            self.down = nn.Sequential(*[nn.MaxPool2d(2)] * layer_depth)

        elif down_layer == "PatchMerging":
            in_channels = [img_ch] + [2*x*rate for x in basic_feature[:layer_depth-1]] + [3*x*rate for x in basic_feature[-layer_depth+1:]]
            out_channels = [x*rate for x in basic_feature]
            self.down = nn.Sequential(*[PatchMerging(x) for x in out_channels[:layer_depth]])
            
        init_DSConvFusion = lambda in_ch, out_ch: \
            nn.Sequential(MultiView_DSConv(in_ch, out_ch, kernel_size, extend_scope, device_id), \
            *[MultiView_DSConv(out_ch, out_ch, kernel_size, extend_scope, device_id)] * (repeat_n-1))

        init_SwinT = lambda in_ch, is_shift: nn.Sequential(*[SwinLayer(in_ch, is_shift)] * repeat_n)
        
        self.dsconvs = nn.Sequential(*[init_DSConvFusion(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, out_channels)])
        self.swins = nn.Sequential(*[init_SwinT(out_ch, i & 1) for i, out_ch in enumerate(out_channels)])

        self.out_conv = nn.Conv2d(rate, output_ch, 1)
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        layer_depth = self.layer_depth

        edc_features = []
        for i in range(layer_depth):
            xi = self.dsconvs[i](x)
            xi = self.swins[i](xi)
            if i < layer_depth - 1:
                edc_features.append(xi)
                x = self.down[i](xi)
            else: x = xi
        # decoder:
        x = self.up(x)

        for i in range(1, layer_depth-1)[::-1]:
            x = torch.cat([x, edc_features[i]], dim=1)
            x = self.dsconvs[2 * (layer_depth-1) - i](x)
            x = self.up(x)
        
        x = torch.cat([x, edc_features[0]], dim=1)
        x = self.dsconvs[2 * (layer_depth-1)](x)
        out = self.out_conv(x)

        return self.sigmoid(out)

class SwinSnake_Dual(nn.Module):
    def __init__(
        self,
        img_ch=3,
        output_ch=1,
        kernel_size=3,
        extend_scope=3,
        layer_depth=2,
        rate=12,
        dim=1,
        device_id="0"
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dim = dim 
        self.layer_depth = layer_depth

        device_id = "cuda:{}".format(device_id)

        basic_feature = [2**x for x in range(layer_depth)] 
        basic_feature += basic_feature[:-1][::-1] # [1, 2, 4, 8, 4, 2, 1]

        in_channels = [img_ch] + [x*rate for x in basic_feature[:layer_depth-1]]

        
        in_channels_2 = {
            2 : [x * rate for x in [7]],
            3 : [x * rate for x in [9, 5]],
            4 : [x * rate for x in [18, 7, 5]],
            5 : [x * rate for x in [36, 14, 7, 5]]
        }

        in_channels += in_channels_2[layer_depth]

        out_channels = [x*rate for x in basic_feature]

        init_DSConvFusion = lambda in_ch, out_ch : MultiView_DSConv(in_ch, out_ch, kernel_size, extend_scope, device_id)
        init_SwinT = lambda in_ch, is_shift: SwinLayer(in_ch, is_shift)

        self.patch_merging = nn.Sequential(*[PatchMerging(x) for x in out_channels[:layer_depth]])
        self.maxpool2d = nn.MaxPool2d(2)
        
        self.dsconvs = nn.Sequential(*[init_DSConvFusion(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, out_channels)])

        self.swins = nn.Sequential(*[init_SwinT(in_ch, i & 1) for i, in_ch in enumerate(in_channels)])
        self.sqzconvs = nn.Sequential(*[Conv(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, out_channels)])

        self.out_conv = nn.Conv2d(rate, output_ch, 1)
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        layer_depth = self.layer_depth

        edc_snake, edc_swin = [], []
        for i in range(layer_depth):
            if i == 0:
                xi = self.dsconvs[i](x)
                edc_snake.append(xi)
                xi = self.maxpool2d(xi)
                x_snake = xi.clone()
                x_swin = xi.clone()
            elif 0 < i < layer_depth - 1:
                # snake
                xi = self.dsconvs[i](x_snake)
                edc_snake.append(xi)
                x_snake = self.maxpool2d(xi)

                # swin
                xi = self.swins[i](x_swin)
                edc_swin.append(xi)
                x_swin = self.patch_merging[i-1](xi)

            else:
                x_snake = self.dsconvs[i](x_snake)
                x_swin = self.swins[i](x_swin) 
        
        # decoder:
        x = self.up(torch.cat([x_snake, x_swin], dim=1))
        
        for i in range(1, layer_depth-1)[::-1]:
            x = torch.cat([x, edc_snake[i], edc_swin[i-1]], dim=1)
            x = self.dsconvs[2 * (layer_depth-1) - i](x)
            x = self.up(x)
            
        x = torch.cat([x, x, edc_snake[0]], dim=1)
        x = self.dsconvs[2 * (layer_depth-1)](x)
        out = self.out_conv(x)

        return self.sigmoid(out)


if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SwinSnake_Alter() #
    size = 400
    summary(model, (1, 3, size, size))

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = MultiView_DSConv(3, 128, 7, 1, True, "cuda:0").to(device)

    # summary(model, (1, 3, 400, 400))