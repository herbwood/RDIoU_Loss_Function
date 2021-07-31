import torch
import torch.nn.functional as F
import torch

def focal_loss(inputs, targets, alpha=-1, gamma=2):
    
    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_pred = (1 - inputs) ** gamma * torch.log(inputs)
    neg_pred = inputs ** gamma * torch.log(1 - inputs)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)

    return loss.sum(axis=1)

#####################Repulsion DIoU loss####################################


def bbox_with_delta(rois, deltas, unnormalize=True):
    # if unnormalize:
    #     std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
    #     mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
    #     deltas = deltas * std_opr + mean_opr

    pred_bbox = bbox_transform_inv_opr(rois, deltas)

    return pred_bbox


# delta : 모델이 예측한 delta 값, shape : (# of positive anchors, 4)
# anchors shape : (# of positive anchors, 4)
# bboxes2 : bboxes1과 비교 대상이 되는 box(gt box, 다른 객체, 혹은 다른 bounding box)
# get_iou : IoU 값 반환 여부 
# epsilon : 분모 0 되는 것을 방지하기 위한 아주 작은 값 
# mask : 최종 loss에 반영할지 여부 
def repulsion_diou_overlap(delta, anchors, bboxes2, get_iou=True, epsilon=5e-10, mask=None):
    
    # anchor와 모델이 예측한 delta 값을 사용하여 변환 
    bboxes1 = bbox_with_delta(anchors, delta)

    if isinstance(bboxes2, tuple):
        bbox2_anchor, bbox2_delta = bboxes2
        bboxes2 = bbox_with_delta(bbox2_anchor, bbox2_delta)

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))

    if rows * cols == 0:
        return dious

    exchange = False

    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    # bboxes1, bboxes2 : [x1, y1, x2, y2]
    # w1, h1 : bboxes1 width, height 
    # w1, h1 shape : (number of positive anchors, 1)
    # w2, h2 : bboxes2 width, height
    # w2, h2 shape : (number of target gt boxes, 1)
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    # pred bbox, gt box area 
    # shape : (numbef of boxes, 1)
    area1 = w1 * h1
    area2 = w2 * h2

    # pred bbox, gt box center point coord 
    # shape : (number of boxes, 1)
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    # inter area coord
    # 두 box가 겹치는 사각형 영역의 x,y 좌표 
    # shape : (number of boxes, 2)
    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    # outer area coord for C
    # 두 box를 감싸는 가장 작은 사각형 영역의 x,y 좌표 
    # shape : (number of boxes, 2)
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    # intersection area
    # inter shape : (number of boxes, 2)
    # inter_area shape : (number of boxes, 1)
    # diagonal distance between pred box and gt box
    # inter_diag shape : (number of boxes, 1)
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)

    # 두 bbox가 겹치는 사각형의 넓이 
    inter_area = inter[:, 0] * inter[:, 1]

    # 두 bbox 사이의 center distance 
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

    # outer area 
    # outer : outer rectangle diagonal x, y coord
    # outer shape : (number of boxes, 2)
    # outer_diag : diagnoal distance of outer rectangle C
    # outer_diag shape : (number of boxes, 1)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)

    # Union area of pred bbox and gt box
    # 두 bbox의 넓이의 합 
    # union shape : (number of boxes, 1)
    union = area1 + area2 - inter_area

    # IoU and center point distance 
    # dious : IoU - (center point distance) 
    # dious shape : (number of boxes, 1)
    iou = inter_area / (union + epsilon)
    center_point_distance = (inter_diag) / (outer_diag + epsilon)
    dious = torch.clamp(dious,min=-1.0,max = 1.0)

    if get_iou:
        dious = iou - center_point_distance
        # mask를 곱해줘, 실제로 loss에 반영할 값에 mask를 곱해줌 
        dious = dious * mask 
        
        return dious

    return center_point_distance * mask


def repulsion_diou_loss(pred_delta, anchors, targets, alpha=0.5, beta=0.5):

    # Repulsion term 1 
    # between positive bbox and positive non target gt boxes 
    # bounding box와 gt box가 아닌 다른 객체와의 distance 
    # Returns :
    # second_gt_indices : second gt indices
    # second_gt_mask : whether iou > 0

    # second_gt_overlaps shape : (# of anchors, # of targets)
    second_gt_overlaps = box_overlap_opr(anchors, targets)

    # 2번째로 높은 iou 값과, index를 구함 
    max_gt_overlaps, gt_assignment = second_gt_overlaps.topk(2, dim=1, sorted=True)
    second_matched_gt_iou, second_gt_indices = max_gt_overlaps[:, 1], gt_assignment[:, 1]
    # second_gt_mask = torch.where(second_matched_gt_iou > 0, 1, 0).cuda()

    # iou 값이 0인 경우 제외하기 위한 mask 
    second_gt_mask = (second_matched_gt_iou > 0).flatten().cuda()


    # Repulsion term 2
    # between bbox and bboxes with different target
    # bounding box와 자신과 다른 객체를 예측하는 bounding box 사이의 distance 
    # Returns :
    # second_bbox_indices : second bbox indices
    # second_bbox_mask : whether iou > 0, whether assigned to same gt box or not 

    # second_bbox_overlaps shape : (anchors, anchors)
    second_bbox_overlaps = box_overlap_opr(anchors, anchors)

    # # 2번째로 높은 iou 값과, index를 구함 
    max_bbox_overlaps, bbox_assignment = second_bbox_overlaps.topk(2, dim=1, sorted=True)
    second_matched_bbox_iou, second_bbox_indices = max_bbox_overlaps[:, 1], bbox_assignment[:, 1]
    
    # iou 값이 0인 경우를 제외하기 위한 mask 
    second_bbox_iou_mask = (second_matched_bbox_iou > 0).flatten()

    # 동일한 gt box를 예측하는 경우를 제외하기 위한 mask 
    second_bbox_gt_mask = torch.all(targets != targets[second_bbox_indices], dim=1).int()

    # 두 mask를 곱해줘 최종 mask 생성 
    second_bbox_mask = second_bbox_iou_mask * second_bbox_gt_mask
    second_bbox_mask = second_bbox_mask.cuda()

    first_gt_mask = torch.ones(anchors.shape[0]).cuda()

    del second_gt_overlaps 
    del second_bbox_overlaps 

    # bbox_gt1_diou : bbox and target gt box iou and center distance, shape : (number of boxes, 1)
    # bbox_gt2_diou : bbox and non target gt box center distance, shape : (number of boxes, 1)
    # bbox_bbox2_diou : bbox and 2nd bbox center distance , shape : (number of boxes, 1)
    bbox_gt1_diou = repulsion_diou_overlap(pred_delta, anchors, targets, get_iou=True, mask=first_gt_mask)
    bbox_gt2_diou = repulsion_diou_overlap(pred_delta, anchors, targets[second_gt_indices], get_iou=False, mask=second_gt_mask)
    bbox_bbox2_diou = repulsion_diou_overlap(pred_delta, anchors, (anchors[second_bbox_indices], pred_delta[second_bbox_indices]), 
                                    get_iou=False, mask=second_bbox_mask)   

    # IoU - center_distance + a * R1 + b * R2
    dious = bbox_gt1_diou + alpha * bbox_gt2_diou + beta * bbox_bbox2_diou
    dious = torch.clamp(dious, min=-3.0, max=3.0) 

    dious = dious.reshape(-1, 1)
    
    # RDIoU = 1 - IoU + center_distance - a * R1 - b * R2
    loss = 1.0 - dious
    loss = loss.sum(axis=1)

    return loss 


def emd_loss_repulsion_diou(p_b0, p_s0, p_b1, p_s1, targets, labels, anchors):

    # pred_delta shape : (# of anchors x 2, 4)
    # pred_score shape : (# of anchors x 2, 1)
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1]) 

    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    anchors = anchors.reshape(-1, 4)

    # 학습에 직접 참여하는 positive/negative label 여부 
    # valid mask : positive/negative label mask(True or False)
    # valid mask shape : (-1, 1)
    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels, config.focal_loss_alpha, config.focal_loss_gamma)

    # positive label에 해당하는 label에 대해서는 localization을 수행함
    # 따라서 negative label은 배제한다
    # fg_masks : positive label mask(True or False)
    fg_masks = (labels > 0).flatten()

    # pred_delta[fg_mask] : positive label에 속하는 delta 값
    # anchor[fg_mask] : positive label에 속하는 anchor
    # targets[fg_mask] : positive label에 속하는 gt box 
    # alpha, beta : balancing parameter 
    localization_loss = repulsion_diou_loss(pred_delta[fg_masks], anchors[fg_masks], targets[fg_masks], alpha=0.3, beta=0.7)

    # final loss : anchor top1+top2 loss, anchor2 top1+top2 loss, ... 
    # loss shape : (anchors x 2, 1) => (anchors, 1) 
    loss = objectness_loss * valid_mask # ignore label이 아닌 애들에 대해서만 loss를 반영함 
    loss[fg_masks] = loss[fg_masks] + localization_loss # positive label인 애들에 대해서만 localization loss를 반영함 
    loss = loss.reshape(-1, 2).sum(axis=1)
    
    # shape : (# of anchors x 2, 1)
    return loss.reshape(-1, 1)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss