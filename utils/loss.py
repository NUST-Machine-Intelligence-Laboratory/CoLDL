import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def regression_loss(x, y):
    # x, y are in shape (N, C)
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1)


def entropy(p):
    return Categorical(probs=p).entropy()


def entropy_loss(logits, reduction='mean'):

    losses = entropy(F.softmax(logits, dim=1))  # (N)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def label_smoothing_cross_entropy(logits, labels, epsilon=0.1, reduction='none'):
    N = logits.size(0)
    C = logits.size(1)
    smoothed_label = torch.full(size=(N, C), fill_value=epsilon / (C - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1 - epsilon)
    if logits.is_cuda:
        smoothed_label = smoothed_label.cuda()
    return cross_entropy(logits, smoothed_label, reduction)


class SmoothingLabelCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self._epsilon = epsilon
        self._reduction = reduction

    def forward(self, logits, labels):
        return label_smoothing_cross_entropy(logits, labels, self._epsilon, self._reduction)


class ScatteredCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self._reduction = reduction

    def forward(self, logits, labels):
        return cross_entropy(logits, labels, self._reduction)
