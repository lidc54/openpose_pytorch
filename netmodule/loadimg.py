# import sys
# sys.path.append('/home/flag54/Downloads/coco/cocoapi-master/PythonAPI/')

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import pylab
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Polygon
import cv2
import torch.utils.data as data
import torch

from netmodule.ground_truth import gt_S_L


def readtest():
    # dataDir = '/home/flag54/Downloads/coco/'
    dataDir = '/media/flag54/54368BA9368B8AA6/DataSet/coco/'
    dataType = 'train2017'
    annFile = '{}annotations/person_keypoints_{}.json'.format(dataDir, dataType)

    # initialize COCO api for person_keypoints_ annotations
    coco = COCO(annFile)
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    catID = coco.getCatIds(catNms=['person'])
    imgID = coco.getImgIds(catIds=catID)
    while True:
        idx = np.random.randint(0, len(imgID))
        single = [304332]  # 432096
        img = coco.loadImgs(imgID[idx])[0]  # single
        img_path = '%strain2017/%s' % (dataDir, img['file_name'])
        data = imread(img_path)
        annID = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(annID)
        get_mask(data, anns)  # anns has several people
        if (len(anns) == 1):
            return data, anns


def get_mask(data, anns):
    height, widht, _ = data.shape
    img = np.zeros((height, widht))
    polygons = []
    for ann in anns:
        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                # polygon
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    polygons.append(poly.astype('int32'))
    cv2.fillPoly(img, polygons, 1)
    return img


def size_adjust(data, row=560, col=656, scales=1):
    """
    adjust data to certain size
    :param data:
    :param row: default 658 (a*16)
    :param col: default 368 (b*16)
    :param scales: to squeeze to this scale,resize again
    :return:
    """
    height, width, *channel = data.shape
    if height / width > row / col:
        ratio = height / row
    else:
        ratio = width / col
    out = cv2.resize(data, (int(width / ratio), int(height / ratio)), interpolation=cv2.INTER_NEAREST)
    h, w, *_ = out.shape
    pad_x = max(0, col - w) // 2
    pad_y = max(0, row - h) // 2

    # padding for this image
    if channel:
        c = channel[0]
        out_ = np.zeros((row, col, c)).astype(data.dtype)
        out_[pad_y:pad_y + min(h, row), pad_x:pad_x + min(w, col), :] = out[0:min(h, row), 0:min(w, col), :]
    else:
        out_ = np.zeros((row, col)).astype(data.dtype)
        out_[pad_y:pad_y + min(h, row), pad_x:pad_x + min(w, col)] = out[0:min(h, row), 0:min(w, col)]

    # resize to certain scale
    out_ = cv2.resize(out_, (col // scales, row // scales), interpolation=cv2.INTER_NEAREST)
    return out_


class coco_pose(data.Dataset):
    """ coco keypoints DataSet Object
    input is image, targets is annotation / confidentce map / PAF

    """

    def __init__(self, dataDir, dataType, annType, signle=False, scales=8):
        """

        :param dataDir:
        :param dataType:
        :param annType:
        :param signle: multi-person or signle person
        :param scales: default is 8,for vgg has three pooling layers
        """
        self.signle = signle
        self.data_dir = dataDir
        self.data_type = dataType
        self.load_data(dataDir, dataType, annType)
        self.scale = scales

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        data, mask, S, L = self.pull_item(id)
        return data, mask, S, L

    def load_data(self, dataDir, dataType, annType):
        annFile = '{}annotations/{}_{}.json'.format(dataDir, annType, dataType)
        self.coco = COCO(annFile)
        catID = self.coco.getCatIds(catNms=['person'])
        imgID = self.coco.getImgIds(catIds=catID)
        if self.signle:
            self.ids = []
            for id in imgID:
                img = self.coco.loadImgs(id)[0]
                annID = self.coco.getAnnIds(imgIds=img['id'])
                anns = self.coco.loadAnns(annID)
                if len(anns) == 1:
                    self.ids.append(id)
        else:
            self.ids = imgID
        print('ok')

    def pull_item(self, img_id):
        img = self.coco.loadImgs([img_id])[0]
        img_path = '%s%s/%s' % (self.data_dir, self.data_type, img['file_name'])
        data = imread(img_path)
        # data = data.transpose((2, 0, 1))
        # data=cv2.imread(img_path)
        annID = self.coco.getAnnIds(imgIds=img['id'])
        anns = self.coco.loadAnns(annID)

        mask = get_mask(data, anns)  # anns has several people
        S, L = gt_S_L(data, anns)

        # adjust their size to certain ones
        S = S.transpose((1, 2, 0))
        L = L.transpose((1, 2, 0))
        data = size_adjust(data)
        S = size_adjust(S, scales=self.scale)  #
        L = size_adjust(L, scales=self.scale)  #
        mask = size_adjust(mask, scales=self.scale)  #

        ss = np.max(S, axis=2)
        ll = np.max(L, axis=2)
        # all is tensor
        data = torch.from_numpy(data.astype('float32')).permute(2, 0, 1)
        mask = torch.from_numpy(mask.astype('float32'))
        S = torch.from_numpy(S.astype('float32')).permute(2, 0, 1)
        L = torch.from_numpy(L.astype('float32')).permute(2, 0, 1)

        return data, mask, S, L


if __name__ == "__main__":
    # readtest()
    # dataDir = '/home/flag54/Downloads/coco/'
    dataDir = '/media/flag54/54368BA9368B8AA6/DataSet/coco/'
    dataType = 'train2017'
    annType = 'person_keypoints'

    test = coco_pose(dataDir, dataType, annType, True)
    x = test[200]
    print('l')
