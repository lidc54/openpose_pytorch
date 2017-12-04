# coding:utf-8

import torch
import numpy as np
from torch.autograd import Variable
import torch.functional as F


class loss_conf(torch.nn.Module):
    """
    St=ρt(F, St−1 , Lt−1 ), ∀t ≥ 2,
    S = max S∗
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


class S_part_conf_map():
    """
     S∗ j is the groundtruth part confidence map
    """

    def __init__(self):
        pass

    def S_jk(self, width, height, keypoints, person, parts=15):
        """
        generate the groundtruth confidence maps S∗
        from the annotated 2D keypoints.
        visible part j for each person k.

        :param width:
        :param height:
        :param keypoints: {1:()}--> parts:(x & y)
        :param person:
        :param parts: keypoints every person
        :return:
        """
        canvas = np.zeros((person, height, width)).astype('float32')
        # multi thread--------------------
        for i in range(person):
            parts_conf = np.zeros((height, width)).astype('float32')
            for j in range(parts):
                coor = keypoints.get(i).get(j)  # alert maybe wrong
                confidence_map = self.gaussian(width, height, coor)
                # max function
                parts_conf = (parts_conf > confidence_map) * parts_conf + \
                             (parts_conf < confidence_map) * confidence_map
            canvas[i] = parts_conf

    def gaussian(self, width, height, coor, sigma=2):
        """
        gaussian like transform,turn coor to DN
        :param width:
        :param height:
        :param coor:
        :param sigma:
        :return:
        """
        ny = np.arange(height).astype('float')
        ny = np.tile(ny, (width, 1)).T
        nx = np.arange(width).astype('float')
        nx = np.tile(nx, (height, 1))
        x, y = coor
        S = np.exp(-(nx - x) ** 2 + (ny - y) ** 2 / (sigma ** 2))
        return S


class L_part_affinity_vector():
    """
     L∗ c is the groundtruth part affinity vector field
    """

    def __init__(self):
        pass

    def L_ck(self, parts):
        for i in range(parts):
            for j in range(parts):
                pass

    def distance(self, length, width, height, coor):
        ny = np.arange(height).astype('float')
        ny = np.tile(ny, (width, 1)).T
        nx = np.arange(width).astype('float')
        nx = np.tile(nx, (height, 1))
        x, y = coor
        vx = (nx - x) / length
        vy = (ny - y) / length
        return vx, vy
