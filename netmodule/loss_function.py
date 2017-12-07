#coding:utf-8
import torch
from torch.autograd import Variable
import torch.functional as F


class loss_conf(torch.nn.Module):
    """
    S=S*,L=L*
    Loss=S+L
    """

    def __init__(self, ground_truth, predicate):
        """
         confidence maps S1 = ρ1 (F)
        """
        super(loss_conf, self).__init__()
        self.gt = ground_truth
        self.p = predicate

    def forward(self, *input):
        pass

class loss_PAF():
    """
    多人：计算所得的paf和候选的肢体的连线做向量的相乘，得到一个Score
    单人：直接Lt-L*
    """
    def __init__(self):
        pass

    def forward(self):
        pass