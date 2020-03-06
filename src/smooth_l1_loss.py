'''
Blog: https://blog.csdn.net/yang_daxia/article/details/91360606
'''
import torch


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


if __name__ == '__main__':
    pred_bboxes = torch.tensor([[15, 18, 47, 60],
                                [50, 50, 90, 100],
                                [70, 80, 120, 145],
                                [130, 160, 250, 280],
                                [25.6, 66.1, 113.3, 147.8]], dtype=torch.float)
    gt_bbox = torch.tensor([[70, 80, 120, 150]], dtype=torch.float)
    print(smooth_l1_loss(pred_bboxes, gt_bbox))
