# coding:utf-8
import torch
from torch.autograd import Variable
import torch.functional as F


#  S=S*,L=L*  --  Loss=S+L

class loss_conf(torch.nn.Module):
    """ confidence map loss & part affinity field
    target:
        1) produce part confidence map loss at stage t
    objective Loss:
        f(S,t) = W(P)*L2((St-S*))
        then integral for every points P in an image
        and every parts J
    """

    def __init__(self):
        """
         confidence maps S1 = ρ1 (F)
        """
        super(loss_conf, self).__init__()

    def forward(self, predication, ground_truth):
        """
        pre shape: batch * parts * h * w
        wp shape: h*w
        :param predication: containing the confidence map
        :param ground_truth: a tuple
            containing the binary mask of people
            ground truth confidence map
        :return:
        """
        wp, sp = ground_truth
        wp=wp.expand(sp.shape)
        loss_S=wp*(predication-sp)**2
        loss_S.data.sum()
        return loss_S


class loss_PAF():
    """ground truth part affinity vector field
        f(L,t) = W(P)*L2((Lt-L*))
        Variable()
    """

    def __init__(self):
        super(loss_PAF,self).__init__()

    def forward(self,prediction,ground_truth):
        """

        :param prediction:
        :param ground_truth: tuple
            containing two dataset:
            binary mask file &
            L ground truth part affinity vector field
        :return:
        """
        wp,lp=ground_truth
        wp=wp.expand_as(lp)
        loss_L=wp*(prediction-lp)**2
        loss_L.data.sum()
        return loss_L

class loss(torch.nn.Module):
    """
    final loss function
    """
    def __init__(self):
        pass

    def forward(self, *input):
        """
        input suppose is a list of Variable of scale
        :param input:
        :return:
        """
        loss=sum(input.data)
        return loss


















