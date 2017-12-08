# coding:utf-8

import torch
import numpy as np

import math
#from netmodule.loadimg import *
from netmodule.config_file import *




class S_part_conf_map():
    """
     S∗ j is the groundtruth part confidence map
    """

    def __init__(self, width, height, keypoints, parts=15):
        self.width = width
        self.height = height
        self.keypoints = keypoints
        self.parts = parts

    def S_jk(self):
        """
        generate the groundtruth confidence maps S∗
        from the annotated 2D keypoints.
        visible part j for each person k.

        :param width:
        :param height:
        :param keypoints: {1:()}--> parts:(x & y)
        :param parts: keypoints every person
        :return:
        """
        width, height, keypoints, parts = self.width, self.height, self.keypoints, self.parts
        sigma = 3 * height / 200
        person = len(keypoints)  # number of people in this photo
        canvas = np.zeros((parts, height, width)).astype('float')
        # multi thread--------------------
        for i in range(person):
            heat_points = keypoints[i]
            for j in range(parts):
                # parts_conf = np.zeros((height, width)).astype('float32')
                *coor, v = heat_points[3 * j:3 * j + 3]
                if v == 0:
                    continue
                confidence_map = self.gaussian(width, height, coor, sigma)
                # max function
                # parts_conf = (parts_conf > confidence_map) * parts_conf + \
                #             (parts_conf < confidence_map) * confidence_map
                canvas[j] = (canvas[j] > confidence_map) * canvas[j] + \
                            (canvas[j] < confidence_map) * confidence_map
        return canvas

    def gaussian(self, width, height, coor, sigma=4):
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
        S = ((nx - x) ** 2 + (ny - y) ** 2) / (sigma ** 2)
        S = np.exp(-S)
        return S.astype('float')


# person
# 先假设一个人的情况，list
# list在openpose中coco是19个点，应该在调用之前做好相应的数据

class L_part_affinity_vector():
    """
     L∗ c is the groundtruth part affinity vector field

    """

    def __init__(self, limbSequence, pose_loc, width, height):
        """

        :param limbSequence: maybe multi lists, deal with it one by one
        :param pose_loc: sepcial for coco 3*19
        :param width:
        :param height:
        """
        self.limbSequence = limbSequence  # if type(limbSequence[0]) == list else [limbSequence]
        self.nlimbSeq = len(limbSequence)  # // 2
        self.canvas = np.zeros((self.nlimbSeq, height, width)).astype('float')
        self.pose = pose_loc
        self.w_canvas = width
        self.h_canvas = height

    def L_ck(self):
        """

        :param pose: (x,y)coor of limb
        :param x: a whiter canvas with certain width & height
        :param y:
        :param h_canvas: the width
        :return:
        """
        # nlimbSeq, h_canvas, w_canvas = self.nlimbSeq, self.heigth, self.width
        # set canvas
        y = np.arange(self.h_canvas).astype('float')
        y = np.tile(y, (self.w_canvas, 1)).T
        x = np.arange(self.w_canvas).astype('float')
        x = np.tile(x, (self.h_canvas, 1))

        # person numbers
        person = len(self.pose)
        for l in range(self.nlimbSeq):
            stickwidth = self.h_canvas / 60  # fixed ::alert!

            # a point in the graph
            limb_a, limb_b = self.limbSequence[l]
            limb_a, limb_b = limb_a - 1, limb_b - 1
            # = limb_sequence[l + 1]

            # loop for all person
            for p in range(person):
                # get their location
                loc = self.pose[p]
                x_a = loc[3 * limb_a]
                y_a = loc[3 * limb_a + 1]
                v_a = loc[3 * limb_a + 2]
                x_b = loc[3 * limb_b]
                y_b = loc[3 * limb_b + 1]
                v_b = loc[3 * limb_b + 2]
                if not v_a or not v_b:
                    continue

                # mid_point
                x_p = (x_a + x_b) / 2
                y_p = (y_a + y_b) / 2

                # cos sin
                if x_b != x_a:
                    angle = math.atan((y_b - y_a) / (x_b - x_a))
                else:
                    angle = math.pi / 2
                cos = math.cos(angle)
                sin = math.sin(angle)
                a_sqrt = (x_a - x_p) ** 2 + (y_a - y_p) ** 2
                b_sqrt = stickwidth ** 2

                # unkownen solution
                A = (x - x_p) * cos + (y - y_p) * sin
                B = (x - x_p) * sin - (y - y_p) * cos
                judge = A * A / a_sqrt + B * B / b_sqrt
                # if (0 <= judge <= 1):
                self.canvas[l] += (judge >= 0) & (judge <= 1)
        return self.canvas


def gt_S_L(data,anno):
    """
    prepaer S for confidence map & L for part affinity
    :return:
    """
    keypoints_list = [i.get('keypoints') for i in anno if 'keypoints' in i]
    num_persons = len(anno)
    height, width, _ = data.shape
    parts_of_coco = 17  # 18 in openpose

    S = S_part_conf_map(width, height, keypoints_list, parts_of_coco)
    s_canvas = S.S_jk()

    #show s_canvas
    ss = np.max(s_canvas, axis=0)
    sss = data * 0
    for i in range(3):
        sss[:, :, i] = (ss * 255) * 0.3 + data[:, :, i] * 0.7
    # plt.imshow(sss)

    print("next is L")

    part_to_limb = coco['keypoints']  # coco18['keypoints']
    limb_sequence = coco['skeleton']  # coco18['limbSequence']
    L = L_part_affinity_vector(limb_sequence, keypoints_list, width, height)
    l_canvas = L.L_ck()

    #show l_canvas
    ss1 = np.max(l_canvas, axis=0)
    sss1 = data * 0
    for i in range(3):
        sss1[:, :, i] = (ss1 * 255) * 0.3 + data[:, :, i] * 0.7
    # plt.imshow(sss1)

    print('okLK')
    return s_canvas,l_canvas
'''
def data_prepare():
    """
    prepare confidence map S & part association L
    :return:
    """
    data, anno = readtest()
    keypoints_list = [i.get('keypoints') for i in anno if 'keypoints' in i]
    num_persons = len(anno)
    height, width, _ = data.shape
    parts_of_coco = 17  # 18 in openpose

    S = S_part_conf_map(width, height, keypoints_list, parts_of_coco)
    s_canvas = S.S_jk()
    ss = np.max(s_canvas, axis=0)
    sss = data * 0
    for i in range(3):
        sss[:, :, i] = (ss * 255) * 0.3 + data[:, :, i] * 0.7

    # plt.imshow(sss)

    print("next is L")
    part_to_limb = coco['keypoints']  # coco18['keypoints']
    limb_sequence = coco['skeleton']  # coco18['limbSequence']
    L = L_part_affinity_vector(limb_sequence, keypoints_list, width, height)
    l_canvas = L.L_ck()
    ss1 = np.max(l_canvas, axis=0)
    sss1 = data * 0
    for i in range(3):
        sss1[:, :, i] = (ss1 * 255) * 0.3 + data[:, :, i] * 0.7
    # plt.imshow(sss1)
    print('okLK')
'''

if __name__ == "__main__":
    data_prepare()
