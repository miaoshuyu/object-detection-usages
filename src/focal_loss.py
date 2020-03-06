'''
Blog:https://www.cnblogs.com/king-lps/p/9497836.html
'''
import torch
import torch.nn.functional as F


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


if __name__ == '__main__':
    loss = focal_loss()

    preds = torch.randn(3, 5)
    target = torch.LongTensor(3).random_(5)

    output = loss(preds, target)
    print(output)
