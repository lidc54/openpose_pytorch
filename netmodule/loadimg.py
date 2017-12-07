# import sys
# sys.path.append('/home/flag54/Downloads/coco/cocoapi-master/PythonAPI/')

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import pylab



def readtest():
    # dataDir = '/home/flag54/Downloads/coco/'
    dataDir = '/media/flag54/54368BA9368B8AA6/DataSet/coco/'
    dataType = 'train2017'
    annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

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
        coco.showAnns(anns)
        get_mask(data,anns[0])#anns has several people
        if (len(anns) == 1):
            return data, anns


def get_mask(self, anns):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    """
    if len(anns) == 0:
        return 0
    if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
        datasetType = 'instances'
    elif 'caption' in anns[0]:
        datasetType = 'captions'
    else:
        raise Exception('datasetType not supported')
    if datasetType == 'instances':
        #ax = plt.gca()
        #ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in anns:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                        polygons.append(Polygon(poly))
                        color.append(c)
                else:
                    # mask
                    t = self.imgs[ann['image_id']]
                    if type(ann['segmentation']['counts']) == list:
                        rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                    else:
                        rle = [ann['segmentation']]
                    m = maskUtils.decode(rle)
                    img = np.ones((m.shape[0], m.shape[1], 3))
                    if ann['iscrowd'] == 1:
                        color_mask = np.array([2.0, 166.0, 101.0]) / 255
                    if ann['iscrowd'] == 0:
                        color_mask = np.random.random((1, 3)).tolist()[0]
                    for i in range(3):
                        img[:, :, i] = color_mask[i]
                    #ax.imshow(np.dstack((img, m * 0.5)))
            if 'keypoints' in ann and type(ann['keypoints']) == list:
                # turn skeleton into zero-based index
                sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton']) - 1
                kp = np.array(ann['keypoints'])
                x = kp[0::3]
                y = kp[1::3]
                v = kp[2::3]
                for sk in sks:
                    if np.all(v[sk] > 0):
                        plt.plot(x[sk], y[sk], linewidth=3, color=c)
                plt.plot(x[v > 0], y[v > 0], 'o', markersize=8, markerfacecolor=c, markeredgecolor='k',
                         markeredgewidth=2)
                plt.plot(x[v > 1], y[v > 1], 'o', markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
        p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
    elif datasetType == 'captions':
        for ann in anns:
            print(ann['caption'])


if __name__ == "__main__":
    readtest()
