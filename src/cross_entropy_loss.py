'''
Blog:https://www.cnblogs.com/marsggbo/p/10401215.html、
https://blog.csdn.net/qq_22210253/article/details/85229988
'''
import torch
import torch.nn.functional as F


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


if __name__ == '__main__':
    preds = torch.randn(12, 3)
    target = torch.ones(12, dtype=torch.long).random_(3)
    print(cross_entropy_loss(preds, target))
    print(cross_entropy_loss2(preds, target))
