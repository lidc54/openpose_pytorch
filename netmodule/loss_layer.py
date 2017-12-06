# coding:utf-8

import torch
import numpy as np
from torch.autograd import Variable
import torch.functional as F
import math


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


# person
# 先假设一个人的情况，list
# list在openpose中coco是19个点，应该在调用之前做好相应的数据

class L_part_affinity_vector():
    """
     L∗ c is the groundtruth part affinity vector field
    """

    def __init__(self, limbSequence):
        self.limbSequence = limbSequence

    def L_ck(self, limbSequence):
        pass

    def distance(self, pose, x, y, h_canvas):
        for l in range(len(self.limbSequence) // 2):
            stickwidth = h_canvas / 60  # fixed ::alert!

            # a point in the graph
            limb_a = self.limbSequence[l]
            limb_b = self.limbSequence[l + 1]
            # get their location
            x_a = pose[3 * limb_a]
            y_a = pose[3 * limb_a + 1]
            v_a = pose[3 * limb_a + 2]
            x_b = pose[3 * limb_b]
            y_b = pose[3 * limb_b + 1]
            v_b = pose[3 * limb_b + 2]

            # mid_point
            x_p = (x_a + x_b) / 2
            y_p = (y_a + y_b) / 2

            # cos sin
            angle = math.atan((y_b - x_b) / (y_a - x_a))
            cos = math.cos(angle)
            sin = math.sin(angle)
            a_sqrt = (x_a - x_p) ** 2 + (y_a - y_p) ** 2
            b_sqrt =stickwidth**2

            # unkownen solution
            A = (x - x_p) * cos + (y - y_p) * sin
            B = (x - x_p) * sin - (y - y_p) * cos
            judge = A * A / a_sqrt + B * B / b_sqrt
            if (0 <= judge <= 1):
                print("gave some value to P")
