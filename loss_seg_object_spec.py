import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .criterion import SegPlusCriterion
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
from misc import nested_tensor_from_tensor_list
import cv2


def dice_loss(inputs, targets, num_masks,epsilon=1e-6):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1) 
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    prob = inputs.sigmoid()
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks



class SegPlusCriterion(nn.Module):
    # in this version, both all masks and logits will be added to compute loss
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, 
                 weight_dict, 
                 losses, 
                 eos_coef=0.1,
                 align_corners=False,
                 
                 ignore_index=255):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()

        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.align_corners = align_corners
        self.loss_map={"masks": self.loss_masks} # ,"contrastive":self.loss_contrastive
        self.ignore_index = ignore_index
        

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        outputs: pred_logits: (bs, n_cls, 1)                       targets: len = bs
                 pred_masks:  (bs, n_cls, H, W)                    targets[0]: 'labels': eg: have the [2, 4] th classes = 2
                 pred: (bs, n_cls, H, W) = pred_logits*pred_masks              'masks':  eg: (2, H, W)
                 aux_outputs: mediate outputs
        """
        # assert "pred_masks" in outputs
        

        # outputs["pred_masks"] = outputs["pred_masks"][:,:-1]

        # for focal loss
        src_masks = outputs # [bs,c_max,h,w]
        src_masks_dice = outputs.clone()
        filtered_src_masks = []
        for i,masks in enumerate(src_masks):
            n_cls = targets[i]["num_class"]
            filtered_src_masks.append(masks[:n_cls]) 
        filtered_src_masks = torch.cat(filtered_src_masks, dim=0) # epsilon(c_i),H,W
        src_masks = filtered_src_masks
        target_masks = self._get_target_mask_binary_cross_entropy(src_masks, targets) # epsilon(c_i),H,W

        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        # for dice loss
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks_dice = outputs
        if src_masks_dice.dim() != 4:
            return {"no_loss": 0}
        src_masks_dice = src_masks_dice[src_idx]
        masks_dice = [t["target_masks"] for t in targets]

        target_masks_dice, valid = nested_tensor_from_tensor_list(masks_dice).decompose()
        target_masks_dice = target_masks_dice.to(src_masks_dice)
        target_masks_dice = target_masks_dice[tgt_idx]

        # upsample predictions to the target size --> for aug_loss
        src_masks_dice = F.interpolate(
            src_masks_dice[:, None], size=target_masks_dice.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks_dice = src_masks_dice[:, 0].flatten(1)

        target_masks_dice = target_masks_dice.flatten(1)
        target_masks_dice = target_masks_dice.view(src_masks_dice.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks_dice, target_masks_dice, num_masks),
        }
   
        return losses
    
    def _get_target_mask_binary_cross_entropy(self, out_masks, targets):
        B, C = out_masks.size()[:2] # b,c+1
        H, W = targets[0]['masks'].size()
        target_masks = []
        for target in targets:
            num_class = target["num_class"]
            mask = target["masks"] # H,W
            for cls in range(num_class):
                target_masks.append((mask== cls).long())
        target_masks = torch.stack(target_masks, dim=0).to(torch.float32)   # epsilon(c_i),H,W
        return target_masks
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = self.loss_map
        
        # if loss=="prompt_tuning":
        #     loss_map.update({"prompt_tuning":self.loss_logits})
        
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        labels = [x['labels'] for x in targets]
        indices_new = []
        for label in labels:
            t = torch.arange(len(label))
            indices_new.append([label, t]) # [[[3,6],[0,1]],[[1],[0]]]
        indices = indices_new
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float
        ).cuda()

        num_masks = torch.clamp(num_masks, min=1).item()

        # if "prompt_pred_masks" in outputs:
        #     # self.losses=["masks"]
        #     self.loss_map.update({"prompt_tuning":self.loss_logits})

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        return losses


class SegLossPlus(nn.Module):
    """ATMLoss.
    """
    def __init__(self,
                 losses=["masks"],
                 mask_weight=20.0,
                 dice_weight=1.0,
                 loss_weight=1.0,
                 align_corners = False,
                 ignore_index=255):
        super(SegLossPlus, self).__init__()
        weight_dict = {"loss_mask": mask_weight, 
                       "loss_dice": dice_weight} 
        self.losses = losses
        self.weight_dict = weight_dict
        self.criterion = SegPlusCriterion(
            weight_dict=weight_dict,
            losses=self.losses,
            align_corners=align_corners
        )
        
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self,
                outputs,
                labels,num_classes
                ):
        """Forward function."""
        
        
        targets = self.prepare_targets(labels,num_classes)
        losses = self.criterion(outputs, targets)
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[k] * self.loss_weight
            else:
                losses.pop(k)
        return losses

    def prepare_targets(self, targets,num_classes):
        new_targets = []
        H,W = targets.size()[1:]
        for targets_per_image in zip(targets,num_classes):
            # gt_cls
            gt_cls = targets_per_image.unique()
            gt_cls = gt_cls[gt_cls != self.ignore_index]
            
            
            masks = []
            for cls in gt_cls:
                masks.append(targets_per_image == cls)
            if len(gt_cls) == 0:
                masks.append(targets_per_image == self.ignore_index)

            masks = torch.stack(masks, dim=0)
            out = {
                "labels": gt_cls,
                "target_masks": masks,
                "masks": targets_per_image,
                "num_class":num_class
            }
            new_targets.append(out)
        return new_targets

