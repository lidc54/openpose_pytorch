import torch
import time
import torch.nn as nn
import torch.utils.data as data
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import cv2

from netmodule.loadimg import coco_pose
from netmodule.net import build_pose
from netmodule.loss_function import *


def train():
    # dataDir = '/home/flag54/Downloads/coco/'
    dataDir = '/media/flag54/54368BA9368B8AA6/DataSet/coco/'
    dataType = 'train2017'
    annType = 'person_keypoints'
    dataset = coco_pose(dataDir, dataType, annType, True)  # true meaning to single

    # some super parameters
    max_iter = 120000
    batch_size = 3
    epoch_size = len(dataset) // batch_size

    # load data
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                  drop_last=True, collate_fn=my_collate)  # pin_memory=True,

    # load net & init
    net = build_pose('train')
    init_net(net)
    # net = torch.nn.DataParallel(net).cuda()
    # net.cuda()
    #cudnn.benchmark = True

    # loss fn
    loss_S = loss_conf()
    loss_L = loss_PAF()

    t0 = time.time()
    lr = 0.9
    momentum = 0.9
    weight_decay = 1e-4

    optimizer = optim.SGD(net.parameters(), lr, momentum=momentum,
                          weight_decay=weight_decay)

    batch_iterator = None
    for iteration in range(max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)

        # snap shot
        img, mask, S, L = next(batch_iterator)
        img = Variable(img)  # .cuda()
        mask = Variable(mask)  # .cuda()
        S = Variable(S)  # .cuda()
        L = Variable(L)  # .cuda()

        conf, paf, out = net(img)
        # conf1, paf1, conf2, paf2, conf3, paf3, conf4, paf4, conf5, paf5, conf6, paf6 = net(img)
        conf = torch.cat(conf, 1)
        paf = torch.cat(paf, 1)

        ls = loss_S(conf, mask, S)
        ll = loss_L(paf, mask, L)
        loss = ls + ll

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('ok')


# init parameters
def xavier(param):
    init.xavier_uniform(param)


def wight_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


# special for this net
def init_net(net):
    # if not resume
    net.vgg.apply(wight_init)
    for conf in net.conf_bra:
        conf.apply(wight_init)
    for paf in net.paf_bra:
        paf.apply(wight_init)


def my_collate(batch):
    """
    to get data stacked, abandon now 2017/12/8
    :param batch: (a tuple) which consist of  data, mask, S, L
                    data is image,mask is segmentation of people,
                    S is ground truth of confidence map
                    L is part affinity vector
    :return:
    """
    data, mask, S, L = [], [], [], []
    for p in batch:
        data_p, mask_p, S_p, L_p = p
        data.append(data_p)
        mask.append(mask_p)
        S.append(S_p)
        L.append(L_p)

    return torch.stack(data, 0), torch.stack(mask, 0), torch.stack(S, 0), torch.stack(L, 0)


if __name__ == "__main__":
    train()
