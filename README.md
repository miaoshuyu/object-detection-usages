# object-detection-usages
the implementation and using examples of object detection usages like, IoU, NMS, soft-NMS, SmoothL1、IoUloss、GIoUloss、 DIoUloss、CIoUloss, cross-entropy、focal-loss、GHM, AP/MAP and so on by Pytorch.



1. **IoU**

   Blog: [https://zhuanlan.zhihu.com/p/47189358](https://zhuanlan.zhihu.com/p/47189358)

   ```
   def get_iou(pred_bbox, gt_bbox):
       '''
       :param pred_bbox: [x1, y1, x2, y2]
       :param gt_bbox:  [x1, y1, x2, y2]
       :return: iou
       '''
   
       ixmin = max(pred_bbox[0], gt_bbox[0])
       iymin = max(pred_bbox[1], gt_bbox[1])
       ixmax = min(pred_bbox[2], gt_bbox[2])
       iymax = min(pred_bbox[3], gt_bbox[3])
       iw = np.maximum(ixmax - ixmin + 1.0, 0.)
       ih = np.maximum(iymax - iymin + 1.0, 0.)
   
       inters = iw * ih
   
       # uni=s1+s2-inters
       uni = (pred_bbox[2] - pred_bbox[0] + 1.0) * (pred_bbox[3] - pred_bbox[1] + 1.0) + \
             (gt_bbox[2] - gt_bbox[0] + 1.0) * (gt_bbox[3] - gt_bbox[1] + 1.0) - inters
   
       iou = inters / uni
   
       return iou
   ```

   

2. **NMS、Soft-NMS、Softer-NMS**

   Blog:[https://zhuanlan.zhihu.com/p/54709759](https://zhuanlan.zhihu.com/p/54709759)

   ```
   def nms(dets, thres):
       '''
       https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
       :param dets:  [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score],,,]
       :param thres: for example 0.5
       :return: the rest ids of dets
       '''
       x1 = dets[:, 0]
       y1 = dets[:, 1]
       x2 = dets[:, 2]
       y2 = dets[:, 3]
       areas = (x2 - x1 + 1) * (y2 - y1 + 1)
       scores = dets[:, 4]
       order = scores.argsort()[::-1]
   
       keep = []
       while order.size > 0:
           i = order[0]
           keep.append(i)
           xx1 = np.maximum(x1[i], x1[order[1:]])
           xx2 = np.minimum(x2[i], x2[order[1:]])
           yy1 = np.maximum(y1[i], y1[order[1:]])
           yy2 = np.minimum(y2[i], y2[order[1:]])
   
           w = np.maximum(xx2 - xx1 + 1.0, 0.0)
           h = np.maximum(yy2 - yy1 + 1.0, 0.0)
   
           inters = w * h
           unis = areas[i] + areas[order[1:]] - inters
           ious = inters / unis
   
           inds = np.where(ious <= thres)[0]  # return the rest boxxes whose iou<=thres
   
           order = order[
               inds + 1]  # for exmaple, [1,0,2,3,4] compare '1', the rest is 0,2 who is the id, then oder id is 1,3
   
       return keep
   ```

   ```
   def soft_nms(dets, iou_thresh=0.3, sigma=0.5, thresh=0.001, method=2):
       '''
       https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py
       :param dets: [[x1, y1, x2, y2, score]，[x1, y1, x2, y2, score]，[x1, y1, x2, y2, score]]
       :param iou_thresh: iou thresh
       :param sigma: std of gaussian
       :param thresh: the last score thresh
       :param method: 1、linear 2、gaussian 3、originl nms
       :return: keep bboxes
       '''
       N = dets.shape[0]  # the size of bboxes
       x1 = dets[:, 0]
       y1 = dets[:, 1]
       x2 = dets[:, 2]
       y2 = dets[:, 3]
       areas = (x2 - x1 + 1) * (y2 - y1 + 1)
   
       for i in range(N):
           temp_box = dets[i, :4]
           temp_score = dets[i, 4]
           temp_area = areas[i]
           pos = i + 1
   
           if i != N - 1:
               maxscore = np.max(dets[:, 4][pos:])
               maxpos = np.argmax(dets[:, 4][pos:])
           else:
               maxscore = dets[:, 4][-1]
               maxpos = -1
   
           if temp_score < maxscore:
               dets[i, :4] = dets[maxpos + i + 1, :4]
               dets[maxpos + i + 1, :4] = temp_box
   
               dets[i, 4] = dets[maxpos + i + 1, 4]
               dets[maxpos + i + 1, 4] = temp_score
   
               areas[i] = areas[maxpos + i + 1]
               areas[maxpos + i + 1] = temp_area
   
           xx1 = np.maximum(x1[i], x1[pos:])
           xx2 = np.minimum(x2[i], x2[pos:])
           yy1 = np.maximum(y1[i], y1[pos:])
           yy2 = np.minimum(y2[i], y2[pos:])
   
           w = np.maximum(xx2 - xx1 + 1.0, 0.)
           h = np.maximum(yy2 - yy1 + 1.0, 0.)
   
           inters = w * h
           ious = inters / (areas[i] + areas[pos:] - inters)
   
           if method == 1:
               weight = np.ones(ious.shape)
               weight[ious > iou_thresh] = weight[ious > iou_thresh] - ious[ious > iou_thresh]
           elif method == 2:
               weight = np.exp(-ious * ious / sigma)
           else:
               weight = np.ones(ious.shape)
               weight[ious > iou_thresh] = 0
   
           dets[pos:, 4] = dets[pos:, 4] * weight
   
       inds = np.argwhere(dets[:, 4] > thresh)
       keep = inds.astype(int)[0]
   
       return keep
   ```

   

3. **The regression loss of object detection：SmoothL1/IoU/GIoU/DIoU/CIoU Loss**

   Blog: https://blog.csdn.net/yang_daxia/article/details/91360606

   https://zhuanlan.zhihu.com/p/104236411

   ```
   def smooth_l1_loss(preds, bboxes, beta=1.0, reduction='mean'):
       '''
       https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/smooth_l1_loss.py
       :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
       :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
       :param beta: float
       :return: loss
       '''
       x_diff = torch.abs(preds - bboxes)
       loss = torch.where(x_diff < beta, 0.5 * x_diff * x_diff / beta, x_diff - 0.5 * beta)
   
       if reduction == 'mean':
           loss = torch.mean(loss)
       elif reduction == 'sum':
           loss = torch.sum(loss)
       else:
           pass
       return loss
   ```

   ```
   def iou_loss(preds, bbox, eps=1e-6, reduction='mean'):
       '''
       https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/iou_loss.py
       :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
       :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
       :return: loss
       '''
       x1 = torch.max(preds[:, 0], bbox[:, 0])
       y1 = torch.max(preds[:, 1], bbox[:, 1])
       x2 = torch.min(preds[:, 2], bbox[:, 2])
       y2 = torch.min(preds[:, 3], bbox[:, 3])
   
       w = (x2 - x1 + 1.0).clamp(0.)
       h = (y2 - y1 + 1.0).clamp(0.)
   
       inters = w * h
   
       uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
               bbox[:, 3] - bbox[:, 1] + 1.0) - inters
   
       ious = (inters / uni).clamp(min=eps)
       loss = -ious.log()
   
       if reduction == 'mean':
           loss = torch.mean(loss)
       elif reduction == 'sum':
           loss = torch.sum(loss)
       else:
           raise NotImplementedError
       return loss
   ```

   ```
   def giou_loss(preds, bbox, eps=1e-7, reduction='mean'):
       '''
      https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36
       :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
       :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
       :return: loss
       '''
       ix1 = torch.max(preds[:, 0], bbox[:, 0])
       iy1 = torch.max(preds[:, 1], bbox[:, 1])
       ix2 = torch.min(preds[:, 2], bbox[:, 2])
       iy2 = torch.min(preds[:, 3], bbox[:, 3])
   
       iw = (ix2 - ix1 + 1.0).clamp(0.)
       ih = (iy2 - iy1 + 1.0).clamp(0.)
   
       # overlap
       inters = iw * ih
   
       # union
       uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
               bbox[:, 3] - bbox[:, 1] + 1.0) - inters + eps
   
       # ious
       ious = inters / uni
   
       ex1 = torch.min(preds[:, 0], bbox[:, 0])
       ey1 = torch.min(preds[:, 1], bbox[:, 1])
       ex2 = torch.max(preds[:, 2], bbox[:, 2])
       ey2 = torch.max(preds[:, 3], bbox[:, 3])
       ew = (ex2 - ex1 + 1.0).clamp(min=0.)
       eh = (ey2 - ey1 + 1.0).clamp(min=0.)
   
       # enclose erea
       enclose = ew * eh + eps
   
       giou = ious - (enclose - uni) / enclose
   
       loss = 1 - giou
   
       if reduction == 'mean':
           loss = torch.mean(loss)
       elif reduction == 'sum':
           loss = torch.sum(loss)
       else:
           raise NotImplementedError
       return loss
   ```

   ```
   def diou_loss(preds, bbox, eps=1e-7, reduction='mean'):
       '''
       https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
       :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
       :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
       :param eps: eps to avoid divide 0
       :param reduction: mean or sum
       :return: diou-loss
       '''
       ix1 = torch.max(preds[:, 0], bbox[:, 0])
       iy1 = torch.max(preds[:, 1], bbox[:, 1])
       ix2 = torch.min(preds[:, 2], bbox[:, 2])
       iy2 = torch.min(preds[:, 3], bbox[:, 3])
   
       iw = (ix2 - ix1 + 1.0).clamp(min=0.)
       ih = (iy2 - iy1 + 1.0).clamp(min=0.)
   
       # overlaps
       inters = iw * ih
   
       # union
       uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
               bbox[:, 3] - bbox[:, 1] + 1.0) - inters
   
       # iou
       iou = inters / (uni + eps)
   
       # inter_diag
       cxpreds = (preds[:, 2] + preds[:, 0]) / 2
       cypreds = (preds[:, 3] + preds[:, 1]) / 2
   
       cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
       cybbox = (bbox[:, 3] + bbox[:, 1]) / 2
   
       inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2
   
       # outer_diag
       ox1 = torch.min(preds[:, 0], bbox[:, 0])
       oy1 = torch.min(preds[:, 1], bbox[:, 1])
       ox2 = torch.max(preds[:, 2], bbox[:, 2])
       oy2 = torch.max(preds[:, 3], bbox[:, 3])
   
       outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2
   
       diou = iou - inter_diag / outer_diag
       diou = torch.clamp(diou, min=-1.0, max=1.0)
   
       diou_loss = 1 - diou
   
       if reduction == 'mean':
           loss = torch.mean(diou_loss)
       elif reduction == 'sum':
           loss = torch.sum(diou_loss)
       else:
           raise NotImplementedError
       return loss
   ```

   ```
   def diou_loss(preds, bbox, eps=1e-7, reduction='mean'):
       '''
       https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
       :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
       :param bbox:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
       :param eps: eps to avoid divide 0
       :param reduction: mean or sum
       :return: diou-loss
       '''
       ix1 = torch.max(preds[:, 0], bbox[:, 0])
       iy1 = torch.max(preds[:, 1], bbox[:, 1])
       ix2 = torch.min(preds[:, 2], bbox[:, 2])
       iy2 = torch.min(preds[:, 3], bbox[:, 3])
   
       iw = (ix2 - ix1 + 1.0).clamp(min=0.)
       ih = (iy2 - iy1 + 1.0).clamp(min=0.)
   
       # overlaps
       inters = iw * ih
   
       # union
       uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
               bbox[:, 3] - bbox[:, 1] + 1.0) - inters
   
       # iou
       iou = inters / (uni + eps)
   
       # inter_diag
       cxpreds = (preds[:, 2] + preds[:, 0]) / 2
       cypreds = (preds[:, 3] + preds[:, 1]) / 2
   
       cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
       cybbox = (bbox[:, 3] + bbox[:, 1]) / 2
   
       inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2
   
       # outer_diag
       ox1 = torch.min(preds[:, 0], bbox[:, 0])
       oy1 = torch.min(preds[:, 1], bbox[:, 1])
       ox2 = torch.max(preds[:, 2], bbox[:, 2])
       oy2 = torch.max(preds[:, 3], bbox[:, 3])
   
       outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2
   
       diou = iou - inter_diag / outer_diag
   
       # calculate v,alpha
       wbbox = bbox[:, 2] - bbox[:, 0] + 1.0
       hbbox = bbox[:, 3] - bbox[:, 1] + 1.0
       wpreds = preds[:, 2] - preds[:, 0] + 1.0
       hpreds = preds[:, 3] - preds[:, 1] + 1.0
       v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
       alpha = v / (1 - iou + v)
       ciou = diou - alpha * v
       ciou = torch.clamp(ciou, min=-1.0, max=1.0)
   
       ciou_loss = 1 - ciou
       if reduction == 'mean':
           loss = torch.mean(ciou_loss)
       elif reduction == 'sum':
           loss = torch.sum(ciou_loss)
       else:
           raise NotImplementedError
       return loss
   ```

   

4. **The classification loss of object detection:cross-entropy/focal-loss/GHM**

   Blog:https://blog.csdn.net/qq_22210253/article/details/85229988 (cross-entropy)
   
   https://www.cnblogs.com/king-lps/p/9497836.html (focal-loss)
   
   https://zhuanlan.zhihu.com/p/80594704 (GHM-loss)
   
   ```
   def cross_entropy_loss(preds, targets):
       '''
       https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/
       :param preds: [N,C]
       :param targets:[N]
       :return:loss
       '''
       loss = F.cross_entropy(preds, targets)
       return loss
   
   
   def cross_entropy_loss2(preds, targets):
       '''
       https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/
       :param preds: [N,C]
       :param targets:[N]
       :return: loss
       '''
       log_softmax = F.log_softmax(preds, dim=1)
       loss = F.nll_loss(log_softmax, targets)
       return loss
   ```
   
   ```
   class focal_loss(torch.nn.Module):
       def __init__(self, gamma=2, alpha=0.25):
           super(focal_loss, self).__init__()
           self.gamma = gamma
           self.alpha = alpha
   
       def forward(self, preds, targets):
           '''
           https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py
           :param preds: [N,C]
           :param targets:[N]
           :return: focal-loss
           '''
           logpt = -F.cross_entropy(preds, targets)
           pt = torch.exp(logpt)
           focal_loss = -self.alpha * (1 - pt) ** self.gamma * logpt
           return focal_loss
   ```
   
   ```
   class GHMC_loss(torch.nn.Module):
       def __init__(self, bins=10, momentum=0, use_sigmiod=True, loss_weight=1.0):
           super(GHMC_loss, self).__init__()
           self.bins = bins
           self.momentum = momentum
           self.edges = [float(x) / bins for x in range(bins + 1)]
           self.edges[-1] += 1e-6
           if momentum > 0:
               self.acc_sum = [0.0 for _ in range(bins)]
           self.use_sigmoid = use_sigmiod
           self.loss_weight = loss_weight
   
       def forward(self, pred, target, label_weight):
           '''
   
           :param pred:[batch_num, class_num]:
           :param target:[batch_num, class_num]:Binary class target for each sample.
           :param label_weight:[batch_num, class_num]: the value is 1 if the sample is valid and 0 if ignored.
           :return: GHMC_Loss
           '''
           if not self.use_sigmoid:
               raise NotImplementedError
           target, label_weight = target.float(), label_weight.float()
           edges = self.edges
           mmt = self.momentum
           weights = torch.zeros_like(pred)
   
           # gradient length
           g = torch.abs(pred.sigmoid().detach() - target)
           valid = label_weight > 0
           total = max(valid.float().sum().item(), 1.0)
           n = 0  # the number of valid bins
   
           for i in range(self.bins):
               inds = (g >= edges[i]) & (g <= edges[i + 1]) & valid
               num_in_bins = inds.sum().item()
               if num_in_bins > 0:
                   if mmt > 0:
                       self.acc_sum[i] = mmt * self.acc_sum[i] + (1 - mmt) * num_in_bins
                       weights[inds] = total / self.acc_sum[i]
                   else:
                       weights[inds] = total / num_in_bins
                   n += 1
   
           if n > 0:
               weights = weights / n
   
           loss = F.binary_cross_entropy_with_logits(pred, target, weights, reduction='sum') / total
   
           return loss * self.loss_weight	
   ```
   
   
   
5. **AP、MAP**

   Blog:https://zhuanlan.zhihu.com/p/48992451
   
   ```
   def voc_ap(rec, prec, use_07_metric=True):
       '''
       https://github.com/amdegroot/ssd.pytorch/blob/5b0b77faa955c1917b0c710d770739ba8fbff9b7/eval.py#L364
       :param prec: [n]
       :param rec: [n]
       :param use_07_metric: use_o7_metric or use_10_metric
       :return: ap
       '''
       if use_07_metric:
           ap = 0.
           for t in np.arange(0., 1.1, 0.1):
               if np.sum(rec >= t) == 0:
                   p = 0
               else:
                   p = np.max(prec[rec >= t])
               ap = ap + p / 11.
       else:
           mrec = np.concatenate(([0.], rec, [1.]))
           mpre = np.concatenate(([0.], prec, [0.]))
           #
           for i in range(mpre.size - 1, 0, -1):
               mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
   
           i = np.where(mrec[1:] != mrec[:-1])[0]
   
           # and sum (\Delta recall) * prec
           ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
   
       return ap
   ```
   
   
   
   
   
