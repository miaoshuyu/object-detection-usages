'''
Blog:https://zhuanlan.zhihu.com/p/80594704
'''

import torch
import torch.nn.functional as F


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


if __name__ == '__main__':
    loss = GHMC_loss()

    preds = torch.randn(3, 5)
    target = torch.LongTensor(3, 1).random_(5)
    print(target)
    target = torch.zeros(3, 5).scatter_(1, target, 1)
    label_weight = torch.ones(3, 5)

    print(preds)
    print(target)
    print(label_weight)

    print(loss(preds, target, label_weight))
