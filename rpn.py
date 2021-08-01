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

class RegionProposalNetwork(nn.Module):
    """
    Implements Region Proposal Network (RPN)

    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature 
            maps.
        head (nn.Module): module that computes the objectness and regression deltas.
        fg_iou_thresh (float): minimum IoU between the anchor and the Gt box so that they can be
            considered as positive during training of the RPN
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss.
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN.
        pre_nms_top_n (Dict[int]): number of proposals to keep before applying NMS. It should
            contain two fields. training and testing, to allow for different values depending 
            on training or evaluation.
        post_nms_top_n (Dict[int]): number of proposals to keep after applying NMS. It should 
            contain two fields: training and testing, to allow for different values depending 
            on training or evaluation.
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals.
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self,
                 anchor_generator,
                 head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0))

        # used during training 
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        # used during testing 
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['testing']
        return self._post_nms_top_n

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            # if gt box is None
            # fill matched_gt_boxes_per_image and labels_per_image with zero values 
            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # get iou value between anchors and gt boxes 
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)

                # -1 : negative 
                # -2 : ignore 
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                # assign label with 0 if negative, and -1 if ignore 
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)
    
