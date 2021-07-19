from collections import OrderedDict

import torch.nn.functional as F
from torch import nn, Tensor

from typing import Tuple, List, Dict, Optional


class ExtraFPNBlock(nn.Module):

    def forward(
        self,
        results: List[Tensor],
        x: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        pass


class FeaturePyramidNetwork(nn.Module):
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super(FeaturePyramidNetwork, self).__init__()

        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        # 1) input conv layer ex) 128 -> 256
        # 2) output conv layer  ex) 256 -> 256
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1) # 1x1 conv
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1) # 3x3 conv 

            # add layers to module list 
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    # Returns i_th inner module output 
    # x : input feature map
    # idx : i_th module index 
    # out : i_th module output 
    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:

        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    # Returns i_th layer module output 
    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:

        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    # x : high resolution feature map to low resolution feature map
    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:

        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        # apply inner block, layer block conv to 
        # lowest resolution(=smallest) feature map
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        # 1) get relatively high resolution feature map and apply inner block conv 
        # 2) interpolate low resolution feature map to high resolution feature map
        # 3) element-wise addition 
        # 4) layer block conv 
        # 5) and add to results list 
        # 6) make it to dict 
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]

            # interpolate small feature map
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")

            # element-wise addition 
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d on top of the last feature map
    """
    def forward(
        self,
        x: List[Tensor],
        y: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))
        return x, names


class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(LastLevelP6P7, self).__init__()

        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)

        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)

        self.use_P5 = in_channels == out_channels

    def forward(
        self,
        p: List[Tensor],
        c: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:

        p5, c5 = p[-1], c[-1]
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        p.extend([p6, p7])
        names.extend(["p6", "p7"])

        return p, names

if __name__ == "__main__":
    # usage example
    import torch

    # [10, 20, 30] : in_channel_list
    # 5 : out_channel 
    m = FeaturePyramidNetwork([10, 20, 30], 5)

    # get some dummy data
    x = OrderedDict()
    x['feat0'] = torch.rand(1, 10, 64, 64)
    x['feat2'] = torch.rand(1, 20, 16, 16)
    x['feat3'] = torch.rand(1, 30, 8, 8)
    
    # compute the FPN on top of x
    output = m(x)
    # returns
    # [('feat0', torch.Size([1, 5, 64, 64])),
    # ('feat2', torch.Size([1, 5, 16, 16])),
    # ('feat3', torch.Size([1, 5, 8, 8]))]
    print([(k, v.shape) for k, v in output.items()])
