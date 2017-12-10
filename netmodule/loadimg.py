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


class coco_pose(data.Dataset):
    """ coco keypoints DataSet Object
    input is image, targets is annotation / confidentce map / PAF

    """

    def __init__(self, dataDir, dataType, annType, signle=False):
        self.signle = signle
        self.data_dir = dataDir
        self.data_type = dataType
        self.load_data(dataDir, dataType, annType)

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
        annID = self.coco.getAnnIds(imgIds=img['id'])
        anns = self.coco.loadAnns(annID)

        mask = get_mask(data, anns)  # anns has several people
        S, L = gt_S_L(data, anns)

        # all is tensor
        data = torch.from_numpy(data.astype('float32')).permute(2, 0, 1)
        mask = torch.from_numpy(mask.astype('float'))
        S = torch.from_numpy(S.astype('float'))
        L = torch.from_numpy(L.astype('float'))

        return data, mask, S, L


if __name__ == "__main__":
    # readtest()
    dataDir = '/home/flag54/Downloads/coco/'
    # dataDir = '/media/flag54/54368BA9368B8AA6/DataSet/coco/'
    dataType = 'train2017'
    annType = 'person_keypoints'

    test = coco_pose(dataDir, dataType, annType, True)
    x = test[262]
    print('l')
