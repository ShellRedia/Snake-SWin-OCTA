# -*- coding: utf-8 -*-
import torch
from torch import nn, cat
from torch.nn.functional import dropout
import os
import einops
from collections import defaultdict
from torchinfo import summary


"""Dynamic Snake Convolution Network"""

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class DSCNet_2D(nn.Module):
    def __init__(
        self,
        img_ch=1,
        output_ch=1,
        kernel_size=5,
        extend_scope=3,
        if_offset=True,
        num_of_feature=256,
        dim=1,
    ):
        """
        Our DSCNet
        :param img_ch: input channel
        :param output_ch: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param num_of_feature: num_of_feature of layer feature
        :param dim:
        """
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.num_of_feature = num_of_feature
        """
        The three contributions proposed in our paper are relatively independent. 
        In order to facilitate everyone to use them separately, 
        we first open source the network part of DSCNet. 
        <dim> is a parameter used by multiple templates, 
        which we will open source in the future ...
        """
        self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
        """
        Here is our framework. Since the target also has non-tubular structure regions, 
        our designed model also incorporates the standard convolution kernel, 
        for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
        """
         
        edc_dct, edcf_dct, dcd_dct = [defaultdict(list) for _ in range(3)]
        
        edc_dct["in_ch"] = [img_ch] + [x * num_of_feature for x in [1, 2, 4, 12, 6, 3]]
        edc_dct["out_ch"] = [x * num_of_feature for x in [1, 2, 4, 8, 4, 2, 1]]

        edcf_dct["in_ch"] = [x * num_of_feature for x in [3, 6, 12, 24, 12]]
        edcf_dct["out_ch"] = [x * num_of_feature for x in [1, 2, 4, 8, 4]]

        dcd_dct["in_ch"] = [x * num_of_feature for x in [6, 3]]
        dcd_dct["out_ch"] = [x * num_of_feature for x in [2, 1]]

        init_DSConv = lambda in_ch, out_ch, axis : DSConv(in_ch, out_ch, kernel_size, extend_scope, axis, if_offset).to(device)

        self.convs = [EncoderConv(in_ch, out_ch).to(device) for in_ch, out_ch in zip(edc_dct["in_ch"], edc_dct["out_ch"])][:-len(dcd_dct["in_ch"])]
        self.convs += [DecoderConv(in_ch, out_ch).to(device) for in_ch, out_ch in zip(dcd_dct["in_ch"], dcd_dct["out_ch"])]
        self.convs = nn.Sequential(*self.convs)

        self.dsconvs_x = [init_DSConv(in_ch, out_ch, 0) for in_ch, out_ch in zip(edc_dct["in_ch"], edc_dct["out_ch"])]
        self.dsconvs_y = [init_DSConv(in_ch, out_ch, 1) for in_ch, out_ch in zip(edc_dct["in_ch"], edc_dct["out_ch"])]
        self.dsconvs_x, self.dsconvs_y = nn.Sequential(*self.dsconvs_x), nn.Sequential(*self.dsconvs_y)

        self.convs_fusion = [EncoderConv(in_ch, out_ch).to(device) for in_ch, out_ch in zip(edcf_dct["in_ch"], edcf_dct["out_ch"])]
        self.convs_fusion += [DecoderConv(in_ch, out_ch).to(device) for in_ch, out_ch in zip(dcd_dct["in_ch"], dcd_dct["out_ch"])]
        self.convs_fusion = nn.Sequential(*self.convs_fusion)

        self.out_conv = nn.Conv2d(self.num_of_feature, output_ch, 1)
        self.maxpooling = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
    
    def multi_view_fusion(self, x, index):
        y = self.convs[index](x)
        y_axis_x = self.dsconvs_x[index](x)
        y_axis_y = self.dsconvs_y[index](x)
        fusion_x = torch.cat([y, y_axis_x, y_axis_y], dim=1)
        return self.convs_fusion[index](fusion_x)

    def multi_view_fusion_res(self, x, x0, index):
        y = self.convs[index](torch.cat([x, x0], dim=1))
        y_axis_x = self.dsconvs_x[index](torch.cat([x, x0], dim=1))
        y_axis_y = self.dsconvs_y[index](torch.cat([x, x0], dim=1))
        fusion_x = torch.cat([y, y_axis_x, y_axis_y], dim=1)
        return self.convs_fusion[index](fusion_x)

    def forward(self, x):
        # encoder:
        x0 = self.multi_view_fusion(x, 0) # block0
        x = self.maxpooling(x0) # block1
        x1 = self.multi_view_fusion(x, 1)
        x = self.maxpooling(x1) # block2
        x2 = self.multi_view_fusion(x, 2)
        x = self.maxpooling(x2) # block3
        x3 = self.multi_view_fusion(x, 3)
        # decoder:
        x = self.up(x3) # block4
        x = self.up(self.multi_view_fusion_res(x, x2, 4)) # block5
        x = self.up(self.multi_view_fusion_res(x, x1, 5)) # block6
        out = self.out_conv(self.multi_view_fusion_res(x, x0, 6)) # block7

        # out = self.dropout(out)
        return self.sigmoid(out)

"""Dynamic Snake Convolution Module"""
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

if __name__=="__main__":
    model = DSCNet_2D().to(device)
    summary(model, (1,1,304,304))
    