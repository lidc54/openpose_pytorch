# import sys
# sys.path.append('/home/flag54/Downloads/coco/cocoapi-master/PythonAPI/')

from pycocotools.coco import COCO
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import pylab

dataDir = '/home/flag54/Downloads/coco/'
dataType = 'val2017'
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir, dataType)

# initialize COCO api for person_keypoints_ annotations
coco = COCO(annFile)
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catID = coco.getCatIds(catNms=['person'])
imgID = coco.getImgIds(catIds=catID)
idx = np.random.randint(0, len(imgID))
img = coco.loadImgs(imgID[idx])[0]
img_path = '%simages/%s' % (dataDir, img['file_name'])
data = imread(img_path)
annID=coco.getAnnIds(imgIds=img['id'])

