# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors

from .metrics import bbox_iou, probiou
from .tal import bbox2dist

from collections import Counter

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability


        print('inside focal loss for class aware')


        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()

def wise_iou_v3_loss(pred_bboxes, target_bboxes):
    """
    Compute the Wise-IoU v3 loss between predicted and target bounding boxes.

    :param pred_bboxes: Predicted bounding boxes, tensor of shape (batch_size, num_boxes, 4)
    :param target_bboxes: Target bounding boxes, tensor of shape (batch_size, num_boxes, 4)
    :return: Wise-IoU v3 loss
    """
    # Calculate intersection
    inter_xmin = torch.max(pred_bboxes[..., 0], target_bboxes[..., 0])
    inter_ymin = torch.max(pred_bboxes[..., 1], target_bboxes[..., 1])
    inter_xmax = torch.min(pred_bboxes[..., 2], target_bboxes[..., 2])
    inter_ymax = torch.min(pred_bboxes[..., 3], target_bboxes[..., 3])
    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)

    # Calculate union
    pred_area = (pred_bboxes[..., 2] - pred_bboxes[..., 0]) * (pred_bboxes[..., 3] - pred_bboxes[..., 1])
    target_area = (target_bboxes[..., 2] - target_bboxes[..., 0]) * (target_bboxes[..., 3] - target_bboxes[..., 1])
    union_area = pred_area + target_area - inter_area

    # Calculate IoU
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    # Calculate Wise-IoU v3 loss
    loss = 1 - iou

    return loss.mean()

class MahalanobisDistance(nn.Module):
    def __init__(self):
        super(MahalanobisDistance, self).__init__()

    def forward(self, x, y):
        """
        Compute the Mahalanobis distance between two sets of points.

        :param x: Tensor of shape (batch_size, num_boxes, 2)
        :param y: Tensor of shape (batch_size, num_boxes, 2)
        :return: Mahalanobis distance
        """
        device = x.device  # Get device of input tensor x
        delta = x - y

        # Ensure delta tensor is on the same device as x and y
        delta = delta.to(device)

        # Compute covariance matrix and its inverse
        cov = torch.matmul(delta.transpose(-1, -2), delta) / delta.size(-2)
        inv_cov = torch.inverse(cov + torch.eye(delta.size(-1), device=device) * 1e-6)  # Add small value for numerical stability

        # Compute Mahalanobis distance
        dist = torch.sqrt(torch.sum(torch.matmul(delta, inv_cov) * delta, dim=-1))

        return dist
class BhattacharyyaDistance(nn.Module):
    def __init__(self):
        super(BhattacharyyaDistance, self).__init__()

    def forward(self, pred_features, target_features):
        """
        Compute Bhattacharyya-like distance between predicted and target feature distributions.

        :param pred_features: Predicted feature distributions, tensor of shape (batch_size, num_boxes, feature_dim)
        :param target_features: Target feature distributions, tensor of shape (batch_size, num_boxes, feature_dim)
        :return: Bhattacharyya-like distance
        """
        # Normalize distributions (assuming softmax for illustration)
        pred_probs = torch.softmax(pred_features, dim=-1)
        target_probs = torch.softmax(target_features, dim=-1)

        # Compute Bhattacharyya coefficient
        b_coeff = torch.sum(torch.sqrt(pred_probs * target_probs), dim=-1)

        # Compute Bhattacharyya distance (negative logarithm)
        b_distance = -torch.log(b_coeff + 1e-6)  # Add epsilon for numerical stability

        return b_distance

def ciou_loss_bhatta(self, pred_bboxes, target_bboxes):
    """
    Compute CIoU (Complete IoU) loss between predicted and target bounding boxes.

    :param pred_bboxes: Predicted bounding boxes, tensor of shape (batch_size, num_boxes, 4)
    :param target_bboxes: Target bounding boxes, tensor of shape (batch_size, num_boxes, 4)
    :return: CIoU loss
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_bboxes[..., 0], pred_bboxes[..., 1], pred_bboxes[..., 2], pred_bboxes[..., 3]
    target_x1, target_y1, target_x2, target_y2 = target_bboxes[..., 0], target_bboxes[..., 1], target_bboxes[..., 2], target_bboxes[..., 3]

    pred_w, pred_h = pred_x2 - pred_x1, pred_y2 - pred_y1
    target_w, target_h = target_x2 - target_x1, target_y2 - target_y1

    pred_cx, pred_cy = (pred_x1 + pred_x2) / 2, (pred_y1 + pred_y2) / 2
    target_cx, target_cy = (target_x1 + target_x2) / 2, (target_y1 + target_y2) / 2

    # Compute IoU
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    pred_area = pred_w * pred_h
    target_area = target_w * target_h
    union_area = pred_area + target_area - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    # Compute aspect ratio
    aspect_ratio = (4 / (torch.pi ** 2)) * (torch.atan(target_w / torch.clamp(target_h, min=1e-6)) - torch.atan(pred_w / torch.clamp(pred_h, min=1e-6))) ** 2
    v = aspect_ratio
    alpha = v / torch.clamp((1 - iou + v), min=1e-6)

    # Compute complete IoU loss
    ciou_loss = 1 - iou + (self.bhattacharyya_distance(pred_bboxes, target_bboxes) / torch.clamp((pred_w ** 2 + pred_h ** 2), min=1e-6)) + alpha * v

    return ciou_loss.mean()


def ciou_loss(pred_bboxes, target_bboxes):
    """
    Compute the Complete IoU (CIoU) loss between predicted and target bounding boxes.

    :param pred_bboxes: Predicted bounding boxes, tensor of shape (batch_size, num_boxes, 4)
    :param target_bboxes: Target bounding boxes, tensor of shape (batch_size, num_boxes, 4)
    :return: CIoU loss
    """
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_bboxes[..., 0], pred_bboxes[..., 1], pred_bboxes[..., 2], pred_bboxes[..., 3]
    target_x1, target_y1, target_x2, target_y2 = target_bboxes[..., 0], target_bboxes[..., 1], target_bboxes[..., 2], target_bboxes[..., 3]

    pred_w, pred_h = pred_x2 - pred_x1, pred_y2 - pred_y1
    target_w, target_h = target_x2 - target_x1, target_y2 - target_y1

    pred_cx, pred_cy = (pred_x1 + pred_x2) / 2, (pred_y1 + pred_y2) / 2
    target_cx, target_cy = (target_x1 + target_x2) / 2, (target_y1 + target_y2) / 2

    # Compute IoU
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    pred_area = pred_w * pred_h
    target_area = target_w * target_h
    union_area = pred_area + target_area - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    ## Compute center distance
    #center_distance = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    #bhattacharyya_distance = BhattacharyyaDistance()
    ##implementation of mahalanobis distance
    ## Compute Mahalanobis distance between centers
    #pred_center = torch.stack([pred_cx, pred_cy], dim=-1)
    #target_center = torch.stack([target_cx, target_cy], dim=-1)
    #center_distance = MahalanobisDistance()(pred_center, target_center)
    #center_distance = bhattacharyya_distance(pred_bboxes, target_bboxes)

    center_distance = wise_iou_v3_loss(pred_bboxes, target_bboxes)
    print('bbox loss calc with wise_iou_v3_loss')

    #b_distance = BhattacharyyaDistance()(pred_features, target_features)
    #center_distance = b_distance 

    # Compute aspect ratio
    aspect_ratio = (4 / (torch.pi ** 2)) * (torch.atan(target_w / torch.clamp(target_h, min=1e-6)) - torch.atan(pred_w / torch.clamp(pred_h, min=1e-6))) ** 2
    v = aspect_ratio
    alpha = v / torch.clamp((1 - iou + v), min=1e-6)

    # Compute complete IoU loss
    ciou_loss = 1 - iou + (center_distance / torch.clamp((pred_w ** 2 + pred_h ** 2), min=1e-6)) + alpha * v

    return ciou_loss.mean()

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        """
        Compute the Adaptive Wing Loss between predictions and targets.

        :param pred: Predicted values, tensor of shape (batch_size, num_boxes, 4)
        :param target: Target values, tensor of shape (batch_size, num_boxes, 4)
        :return: Adaptive Wing Loss
        """
        diff = pred - target
        c = self.theta * (1 + torch.log(1 + torch.tensor(self.omega / self.epsilon)))
        loss = torch.where(
            torch.abs(diff) < self.theta,
            self.omega * torch.log(1 + torch.abs(diff / self.epsilon)),
            torch.abs(diff) - c
        )
        return torch.mean(torch.pow(1 + torch.pow(diff / self.epsilon, 2), self.alpha - target) * loss)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.awing_loss = AdaptiveWingLoss()
        #self.bhattacharyya_distance = BhattacharyyaDistance()

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        #iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        #loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        #print('without wise iou v3 loss')

        ##print("inside wise iou v3 loss")
        #loss_iou = wise_iou_v3_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        #wise_iou_loss_value = wise_iou_v3_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        #loss_iou = wise_iou_loss_value
        #print('with wise iou v3 loss')

        ## Compute CIoU loss
        ciou_loss_value = ciou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ciou_loss_value
        #print("CIoU loss impl")
        #print("CIoU loss with mahalanobis impl")

        #ciou_loss_value = ciou_loss_bhatta(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        #loss_iou = ciou_loss_value

        ## Compute Adaptive Wing Loss
        #awing_loss_value = self.awing_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        #loss_iou = awing_loss_value
        #print("adaptve wing loss")

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, use_dfl)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


import torch
import torch.nn as nn
import torch.nn.functional as F

class VarifocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(VarifocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target, labels):
        # Compute binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Compute the pt term (probabilities of true labels)
        pt = torch.exp(-bce_loss)

        # Compute the varifocal loss
        varifocal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # Create a mask with the same shape as pred
        mask = torch.zeros_like(pred)

        # Ensure labels are long and reshaped properly
        labels = labels.long()

        # Scatter the mask with labels
        for i in range(mask.size(0)):  # Iterate over the batch size
            mask[i].scatter_(0, labels[i], 1)

        # Apply the mask to the varifocal loss
        varifocal_loss *= mask

        return torch.mean(varifocal_loss)


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)


    def calculate_iou(pred_boxes, true_boxes):
    # Calculate intersection
        inter_x1 = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], true_boxes[:, 3])
    
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    
        union_area = pred_area + true_area - inter_area
    
    # Calculate IoU
        iou = inter_area / torch.clamp(union_area, min=1e-6)
    
        return iou

    def calculate_center(boxes):
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        return centers

    def calculate_euclidean_distance(center1, center2):
        return torch.norm(center1 - center2, dim=1)

    def calculate_enclosing_box_diagonal(pred_boxes, true_boxes):
        enclosing_x1 = torch.min(pred_boxes[:, 0], true_boxes[:, 0])
        enclosing_y1 = torch.min(pred_boxes[:, 1], true_boxes[:, 1])
        enclosing_x2 = torch.max(pred_boxes[:, 2], true_boxes[:, 2])
        enclosing_y2 = torch.max(pred_boxes[:, 3], true_boxes[:, 3])
    
        enclosing_diagonal = torch.norm(torch.stack([enclosing_x2 - enclosing_x1, enclosing_y2 - enclosing_y1], dim=1), dim=1)
    
        return enclosing_diagonal

    def calculate_aspect_ratio_term(pred_boxes, true_boxes):
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        true_w = true_boxes[:, 2] - true_boxes[:, 0]
        true_h = true_boxes[:, 3] - true_boxes[:, 1]
    
        aspect_ratio_pred = pred_w / torch.clamp(pred_h, min=1e-6)
        aspect_ratio_true = true_w / torch.clamp(true_h, min=1e-6)
    
        v = (4 / (3.141592653589793**2)) * torch.pow(torch.atan(aspect_ratio_true) - torch.atan(aspect_ratio_pred), 2)
    
        return v


    def wise_iou_v3_loss(pred_boxes, true_boxes):
    # Calculate IoU
        iou = calculate_iou(pred_boxes, true_boxes)
    
    # Calculate distance term (DIoU)
        center_pred = calculate_center(pred_boxes)
        center_true = calculate_center(true_boxes)
        distance = calculate_euclidean_distance(center_pred, center_true)
        c = calculate_enclosing_box_diagonal(pred_boxes, true_boxes)
        diou_term = distance / c
    
    # Calculate aspect ratio term (CIoU)
        aspect_ratio_term = calculate_aspect_ratio_term(pred_boxes, true_boxes)
    
    # Dynamic weights (these can be tuned)
        w1, w2, w3 = 1.0, 1.0, 1.0
    
    # Calculate Wise-IoU v3 loss
        wise_iou_v3 = iou - w1 * diou_term - w2 * aspect_ratio_term * iou - w3 * (aspect_ratio_term)
    
        return 1 - wise_iou_v3


    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        #print('pred gt_labels', gt_labels)
        ##print('pred gt_labels', gt_labels[2])
        #print('shape of gt_labels', gt_labels.shape)

        ### Determine the device from target_scores
        device = targets.device

        #### Count class frequencies from target scores
        #unique_classes, class_counts = torch.unique(targets, return_counts=True)
        #print('class counts',class_counts)       
        #class_frequencies = class_counts.float() / targets.size(0)
        #print('class frequencies',class_frequencies)
        #print('unique_classes',unique_classes)


        #print('targets before count',targets)

        samples_per_cls = Counter(gt_labels.view(-1).tolist())
        ###samples_per_cls = Counter(targets.view(-1).tolist())
        #print('samples_per_cls counted',samples_per_cls)   

        # Total number of samples
        total_samples = sum(samples_per_cls.values())
        #print('total samples',total_samples)

        ## Class coefficient for each class
        #eta = 4
        eta = 8
        #eta = 12
        #eta = 6
        class_coeff = 1 - torch.pow(torch.Tensor(list(samples_per_cls.values())) / total_samples, eta)


        ### Calculate class weights as the inverse of the class frequencies
        #class_weights = 1.0 / class_frequencies
        ### Normalize class weights so that the mean weight is 1
        #class_weights = class_weights / class_weights.mean()
        #print('class weights',class_weights)

        self.num_classes = self.nc 
        # weights for each class
        class_weights = (1 - torch.Tensor(list(samples_per_cls.values())) / total_samples) / class_coeff
        # Normalize the class weights
        class_weights = class_weights / torch.sum(class_weights) * self.num_classes

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = class_weights.to(device)
        #print('class weights after normalization and device change',class_weights)

        # class-aware loss
        #class_aware_loss = torch.sum(class_weights * cross_entropy_loss)

        ##num_classes = self.nc
        ### Convert class weights to a tensor
        ##class_weights_tensor = torch.ones(num_classes, device=device)
        ##print('class weights tensor',class_weights_tensor)
        ##print('unique_classes tensor',unique_classes)
        ##print('class weights tensor shape',class_weights_tensor.shape)
        ##print('unique_classes tensor shape',unique_classes.shape)
        ##class_weights_tensor[unique_classes.long()] = class_weights.to(device) 
        ##print('class weights tensor',class_weights_tensor)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        #print('number of classes',self.nc)
        #print('shape of target_scores', target_scores.shape)
        #print('shape of predicted_scores', pred_scores.shape)
        ###print('shape of target_labels', target_labels.shape)
        ###print('shape of gt_labels', gt_labels.shape)
        #print('shape of targets', targets.shape)


        ## Cls loss
        #loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way

        #verifocalloss_function = VarifocalLoss(alpha=0.25, gamma=2.0)
        #target_labels = gt_labels
        #loss[1] = verifocalloss_function(pred_scores, target_scores, target_labels)
        #print("implementation of verifocal loss")    
     
        ##original is bce loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        #print('loss after bce',loss[1])

        ### class-aware loss
        #class_aware_loss_new = torch.sum(class_weights * loss[1])
        #print('class_aware_loss implemented',class_aware_loss_new)

        #loss[1] = class_aware_loss_new
        #print('loss[1] after class aware term',loss[1])

        ###Try to implement class aware loss funciton here
        #print('loss in cls inside loss py func',loss[1])

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

           
        
        #loss[3] = wise_iou_v3_loss(pred_bboxes, target_bboxes)
        #print('wise iou v3 loss',loss[3])
        #loss[0] = loss[3]
        #loss[0] *= self.hyp.box  # box gain

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        #print("implementation of class aware loss")
        ##implementing class aware loss function
        loss[1] = torch.sum(class_weights * loss[1])
        loss[0] = torch.sum(class_weights * loss[0])
        loss[2] = torch.sum(class_weights * loss[2])

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        """
        Initializes v8OBBLoss with model, assigner, and rotated bbox loss.

        Note model must be de-paralleled.
        """
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)
