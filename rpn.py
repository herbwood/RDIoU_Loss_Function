import torch
from torch.nn import functional as F
from torch import nn, Tensor

import torchvision 
from torchvision.ops import boxes as box_ops

import _utils as det_utils
from image_list import ImageList

from typing import List, Optional, Dict, Tuple 


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted 
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Args:
            x (List[Tensor]): feature pyramid network outputs 
        
        Returns:
            logits, bbox_reg (Tuple[List[Tensor], List[Tensor]]): the output from the RPN head
                class scores and bbox regressor for feature maps with different resolutions 
        """
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits)
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    """
    Reshape for layer output

    Args: 
        layer (Tensor): class scores or bbox regressors
        N (int): batch size 
        A (int): number of anchors per location
        C (int): channel size 
        H (int): height
        W (int): width 

    Returns:
        layer (Tensor): reshaped layer 
    """
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer 


def concat_box_prediction_layers(box_cls, box_regression):
    # type : (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the 
    # same format as the labels. Note that the labels are computed for 
    # all feature levels concatenated, so we keep the same representation 
    # for the objectness and the box_regression 
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape 
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4 # number of anchors 
        C = AxC // A 
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression 
