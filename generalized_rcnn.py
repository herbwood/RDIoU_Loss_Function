from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union


class GenralizedRCNN(nn.Module):
    """
    Main class for Genralized R-CNN

    Args:
        backbone (nn.Module)
        rpn (nn.Module)
        roi_heads (nn.Module) : takes the feature + the proposals from the RPN and computes 
            detections / masks from it. 
        transform (nn.Module) : performs the data transformation from the inputs to feed into
            the model 
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GenralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone 
        self.rpn = rpn
        self.roi_heads = roi_heads 

    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses. 
                During testing, it returns list[BoxList] contains additional fields
                like 'scores', 'labels', 'mask' (for Mask R-CNN models). 
        """
        if self.training and targets is None:
            raise ValueError("In training model, targets should be passed")
        if self.training:
            assert targets is not None

            # GT box shape,dtype check 
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected target bxes to be a tensor"
                                         f"of shape [N, 4], got {boxes.shape}.")

                    else:
                        raise ValueError(f"Expected target boxes to be of type" 
                                         f"Tensor, got {type(boxes)}.")

        # add original image sizes
        original_image_sizes : List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:] # (height, width)
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]

                # degenerate boxes are boxes with x2y2 valeus smaller than x1y1
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(f"All bounding boxes should have positive height and width."
                                     f" Found invalid box {degen_bb} for target at index {target_idx}")

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict(['0', features])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections 

        