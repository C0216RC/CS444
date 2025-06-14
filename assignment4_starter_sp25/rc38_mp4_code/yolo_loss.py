import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter + 1e-10)
    return iou


class YoloLoss(nn.Module):

    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord  
        self.l_noobj = l_noobj  
        
        self._epoch = 0
        
        self._confidence_boost_initial = 10.0
        self._confidence_boost_decay = 0.7  
        self._min_confidence_boost = 3.0 

        self._coord_scale_initial = 2.0
        self._coord_scale_decay = 0.85
        self._min_coord_scale = 1.0

        self._class_suppression_initial = 0.5 
        self._class_suppression_final = 1.0

        self._warmup_epochs = 3  
        self._use_hard_examples_mining = True  
        self._confidence_threshold = 0.3 

    def update_epoch(self, epoch):
        self._epoch = epoch

    def get_confidence_boost(self):
        if self._epoch < self._warmup_epochs:
            return self._confidence_boost_initial
        else:
            decay_epochs = self._epoch - self._warmup_epochs
            decay_factor = self._confidence_boost_decay ** decay_epochs
            boost = max(self._min_confidence_boost, 
                        self._confidence_boost_initial * decay_factor)
            return boost

    def get_coord_scale(self):
        if self._epoch < self._warmup_epochs:
            return self._coord_scale_initial
        else:
            decay_epochs = self._epoch - self._warmup_epochs
            decay_factor = self._coord_scale_decay ** decay_epochs
            scale = max(self._min_coord_scale, 
                         self._coord_scale_initial * decay_factor)
            return scale

    def get_class_weight(self):
        if self._epoch < 5:
            return self._class_suppression_initial
        elif self._epoch < 10:
            progress = (self._epoch - 5) / 5.0
            return self._class_suppression_initial + progress * (
                self._class_suppression_final - self._class_suppression_initial)
        else:
            return self._class_suppression_final

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        x = boxes[:, 0].unsqueeze(1) / self.S
        y = boxes[:, 1].unsqueeze(1) / self.S
        w = boxes[:, 2].unsqueeze(1)
        h = boxes[:, 3].unsqueeze(1)
        w = torch.clamp(w, min=1e-10)
        h = torch.clamp(h, min=1e-10)
        

        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        
        x1 = torch.clamp(x1, min=0.0, max=1.0)
        y1 = torch.clamp(y1, min=0.0, max=1.0)
        x2 = torch.clamp(x2, min=0.0, max=1.0)
        y2 = torch.clamp(y2, min=0.0, max=1.0)
        
        return torch.cat([x1, y1, x2, y2], dim=1)

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : (list) [(tensor) size (-1, 5)]  
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions
        """
        box_target_xyxy = self.xywh2xyxy(box_target)

        ious = torch.zeros(box_target.size(0), self.B, device=box_target.device)
        
        for b in range(self.B):

            pred_box_xyxy = self.xywh2xyxy(pred_box_list[b][:, :4])

            iou_matrix = compute_iou(pred_box_xyxy, box_target_xyxy)

            ious[:, b] = torch.diagonal(iou_matrix)
        best_ious, best_box_indices = torch.max(ious, dim=1)

        best_boxes = torch.zeros_like(pred_box_list[0])

        for i in range(best_box_indices.size(0)):
            box_idx = best_box_indices[i]
            best_boxes[i] = pred_box_list[box_idx][i]

        if self._epoch < 5:

            if self._epoch == 0:
                boost = 0.5
            else:
                boost = 0.5 - self._epoch * 0.05 
 
            best_ious = torch.clamp(best_ious + boost, max=0.95)
        
        return best_ious.unsqueeze(1), best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
  
        has_object_mask = has_object_map.unsqueeze(-1).expand_as(classes_target)

        pred_masked = classes_pred[has_object_mask]
        target_masked = classes_target[has_object_mask]

        if pred_masked.size(0) == 0:
            return torch.tensor(0.0, device=classes_pred.device)

        class_weight = self.get_class_weight()

        if self._epoch < 10:

            loss = F.binary_cross_entropy(
                torch.clamp(pred_masked, min=1e-7, max=1.0-1e-7),
                target_masked,
                reduction='sum'
            )
        else:

            loss = F.mse_loss(pred_masked, target_masked, reduction='sum')

        return loss * class_weight

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar
        """

        no_object_mask = ~has_object_map
        no_obj_loss = 0

        for pred_boxes in pred_boxes_list:
            pred_conf = pred_boxes[..., 4]
            
            no_obj_conf = pred_conf[no_object_mask]
            
            if self._epoch < self._warmup_epochs:

                target_conf = torch.zeros_like(no_obj_conf) + 0.05
            else:

                target_conf = torch.zeros_like(no_obj_conf)
            
            no_obj_loss += F.mse_loss(no_obj_conf, target_conf, reduction='sum')
        
        return no_obj_loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar
        """
        boost_factor = self.get_confidence_boost()

        contain_loss = F.mse_loss(box_pred_conf, box_target_conf, reduction='none')

        if self._use_hard_examples_mining and self._epoch >= 5:
 
            diff = torch.abs(box_pred_conf - box_target_conf)
            
            hard_indices = diff.flatten() > self._confidence_threshold
            
            if hard_indices.sum() > 0:
                contain_loss[hard_indices] = contain_loss[hard_indices] * 2.0
        return (contain_loss * boost_factor).sum()

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar
        """

        pred_x = box_pred_response[:, 0]
        pred_y = box_pred_response[:, 1]
        pred_w = box_pred_response[:, 2]
        pred_h = box_pred_response[:, 3]
        
        target_x = box_target_response[:, 0]
        target_y = box_target_response[:, 1]
        target_w = box_target_response[:, 2]
        target_h = box_target_response[:, 3]
        
        pred_w = torch.abs(pred_w) + 1e-6
        pred_h = torch.abs(pred_h) + 1e-6
        target_w = torch.abs(target_w) + 1e-6
        target_h = torch.abs(target_h) + 1e-6
        
        loss_x = F.mse_loss(pred_x, target_x, reduction='sum')
        loss_y = F.mse_loss(pred_y, target_y, reduction='sum')

        loss_w = F.mse_loss(
            torch.sqrt(pred_w),
            torch.sqrt(target_w),
            reduction='sum'
        )
        loss_h = F.mse_loss(
            torch.sqrt(pred_h),
            torch.sqrt(target_h),
            reduction='sum'
        )
        

        coord_scale = self.get_coord_scale()
        
        reg_loss = (loss_x + loss_y + loss_w + loss_h) * coord_scale
        
        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) where:  
                            N - batch_size
                            S - width/height of network output grid
                            B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        batch_size = N

        pred_boxes_list = []
        for b in range(self.B):
            pred_boxes_list.append(pred_tensor[..., b*5:b*5+5])
        
        pred_cls = pred_tensor[..., self.B*5:]

        cls_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)

        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)

        target_boxes_reshape = target_boxes.view(-1, 4)
        
        pred_boxes_reshape_list = []
        for b in range(self.B):
            pred_boxes_reshape_list.append(pred_boxes_list[b].contiguous().view(-1, 5))

        has_obj_mask = has_object_map.view(-1)
 
        if has_obj_mask.sum() == 0:
            total_loss = (no_obj_loss + cls_loss) / batch_size
            return {
                'total_loss': total_loss,
                'reg_loss': torch.tensor(0.0, device=pred_tensor.device),
                'containing_obj_loss': torch.tensor(0.0, device=pred_tensor.device),
                'no_obj_loss': no_obj_loss / batch_size,
                'cls_loss': cls_loss / batch_size
            }
        
        target_boxes_obj = target_boxes_reshape[has_obj_mask]
        
        pred_boxes_obj_list = []
        for b in range(self.B):
            pred_boxes_obj_list.append(pred_boxes_reshape_list[b][has_obj_mask])
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_obj_list, target_boxes_obj)

        reg_loss = self.get_regression_loss(best_boxes[:, :4], target_boxes_obj)

        containing_obj_loss = self.get_contain_conf_loss(best_boxes[:, 4].unsqueeze(1), best_ious)
        reg_loss = self.l_coord * reg_loss
        no_obj_loss = self.l_noobj * no_obj_loss
        reg_loss = reg_loss / batch_size
        containing_obj_loss = containing_obj_loss / batch_size
        no_obj_loss = no_obj_loss / batch_size
        cls_loss = cls_loss / batch_size

        total_loss = reg_loss + containing_obj_loss + no_obj_loss + cls_loss

        loss_dict = {
            'total_loss': total_loss,
            'reg_loss': reg_loss,
            'containing_obj_loss': containing_obj_loss,
            'no_obj_loss': no_obj_loss,
            'cls_loss': cls_loss,
        }
        
        return loss_dict