import sys
sys.path.append('/media/flag54/54368BA9368B8AA6/DataSet/coco/cocoapi-master/PythonAPI/')

from pycocotools.coco import COCO
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import pylab

dataDir = '/media/flag54/54368BA9368B8AA6/DataSet/coco/'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

# initialize COCO api for instance annotations
coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# imgIds = coco.getImgIds(catIds=catIds );
imgIds = coco.getImgIds(imgIds=[324158])
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

# load and display image
# I = imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()
