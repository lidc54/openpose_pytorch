import torch
import time
import torch.nn as nn
import torch.utils.data as data
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import cv2
import os
from math import isnan
import torchvision.models as models

import numpy as np
from netmodule.loadimg import coco_pose
from netmodule.net import build_pose
from netmodule.loss_function import *
from netmodule.config_file import path_vgg


def train():
    t0 = time.time()
    # dataDir = '/home/flag54/Downloads/coco/'
    # dataDir = '/media/flag54/54368BA9368B8AA6/DataSet/coco/'
    dataDir = '/home/jinx/PycharmProjects/pose-estimation/dataset/coco/'
    dataType = 'train2017'
    annType = 'person_keypoints'

    dataset = coco_pose(dataDir, dataType, annType, True)  # true meaning to single

    # to check the mean loss of the span
    mean_loss = []
    span = 50
    # some super parameters
    max_iter = 120000
    batch_size = 8
    epoch_size = len(dataset) // batch_size
    snap_shot = 300  # save parameters every times
    parents = '../'
    path_the_net = parents + 'pose__{}__model.pth'
    resume = True
    iteration = 0

    # load data
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, collate_fn=my_collate)  # pin_memory=True,

    # load net & init
    net = build_pose('train')

    if resume:
        now_iter = load_parameters(net, path_the_net)
        if now_iter:
            iteration = now_iter
    # net = torch.nn.DataParallel(net).cuda()
    net.cuda()
    cudnn.benchmark = True

    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-3
    # params = get_param(net, lr)
    # loss fn

    optimizer = optim.SGD([{'params': net.conf_bra.parameters()},
                           {'params': net.paf_bra.parameters()},
                           {'params': net.vgg[23].parameters()},
                           {'params': net.vgg[25].parameters()}],
                          lr=lr, momentum=momentum, weight_decay=weight_decay)

    loss_pose = lossPose()

    print('prepare time is: ', time.time() - t0)
    batch_iterator = None

    while iteration < max_iter:
        t0 = time.time()

        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)

            # lr adjust
            for param_lr in optimizer.param_groups:
                param_lr['lr'] /= 2

            # save parameters  snap shot
            if not resume and iteration > 0:
                torch.save({'iter': iteration, 'net_state': net.state_dict()}, path_the_net.format(iteration))
            resume = False

        img, mask, S, L = next(batch_iterator)
        img = Variable(img, requires_grad=True).cuda()

        conf, paf, out = net(img, mask)
        conf = torch.cat(conf, 1)
        paf = torch.cat(paf, 1)

        ls = loss_pose(conf, mask, S)
        ll = loss_pose(paf, mask, L)

        loss = ls + ll

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # mean loss of span
        if loss.data[0] and not isnan(loss.data[0]):
            mean_loss.append(loss.data[0])
        if len(mean_loss) >= span:
            mean_loss.pop(0)
        print('{:6d} ,with loss is {:.3f} , and time is {:.3f} average is {:.3f}'
              .format(iteration, loss.data[0], time.time() - t0, np.mean(mean_loss)))
        iteration += 1


def get_param(model, lr):
    # get param for optims,bias and weight have two lr
    lr1 = []
    lr2 = []
    param_dict = dict(model.module.named_paramters())
    for key, value in param_dict.items():
        if 'bias' in key:
            lr2.append(value)
        else:
            lr1.append(value)
    params = [{'params': lr1, 'lr': lr},
              {'params': lr2, 'lr': lr * 2}]
    return params


# init parameters
def xavier(param):
    init.xavier_uniform(param)


def wight_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        # m.bias.data.zero_()+0.1
        init.constant(m.bias, 0.1)


# special for this net
def init_net(net):
    # if not resume
    net.vgg.apply(wight_init)  # init vgg with xavier
    vgg_init(net.vgg)
    for conf in net.conf_bra:
        conf.apply(wight_init)
    for paf in net.paf_bra:
        paf.apply(wight_init)


def vgg_init(net):
    # download parameters from vgg net
    param_num = 10
    if not path_vgg:
        vgg_19 = models.vgg19(pretrained=True)
        param_num *= 2
    else:
        vgg_19 = torch.load(path_vgg)
        param_num *= 2
    model_dict = net.state_dict()
    # delete last two layer in net 23 and 25
    '''
    last_drop = ['23', '25']
    vgg_dict = {k.split('features.')[-1]: v for k, v in vgg_19.items()
                if k.split('features.')[-1] in model_dict and
                k.split('.')[1] not in last_drop}
    '''

    vgg_19_key = list(vgg_19.keys())
    model_key = list(model_dict.keys())
    from collections import OrderedDict
    vgg_dict = OrderedDict()
    for i in range(param_num):
        vgg_dict[model_key[i]] = vgg_19[vgg_19_key[i]]

    model_dict.update(vgg_dict)
    net.load_state_dict(model_dict)
    length = len(net)
    for i, layer in enumerate(net):
        if i <= length - 5:
            for j in layer.parameters():
                j.requires_grad = False


def load_parameters(net, path):
    '''init the net or load existing parameters'''
    # find existing pth
    parents = path.split('pose_')[0]
    bigger = []
    for file in os.listdir(parents):
        if '.pth' in file:
            file = file.split('__')
            if len(file) >= 2:
                bigger.append(int(file[1]))
    if bigger:
        save_data = torch.load(path.format(max(bigger)))
        net.load_state_dict(save_data['net_state'])
        now_iter = save_data['iter']
        return now_iter
    else:
        init_net(net)


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
