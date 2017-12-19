'''
this function consist several stages of the networks
1.vgg19
2.stage_i
'''
import torch
import torch.nn as nn

# openpose use first 10 layers of vgg19
vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512]  # , 512, 512, 'M', 512, 512, 512, 512, 'M'
stage = [[3, 3, 3, 1, 1], [7, 7, 7, 7, 7, 1, 1]]


def make_layer(cfg):
    """
    make vgg base net acoording to cfg
    :param cfg: input channels for
    :return:list of layers
    """
    layer = []
    in_channels = 3
    for i in cfg:
        if i == 'M':
            layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, i, kernel_size=3, padding=1)
            layer += [conv2d, nn.ReLU(inplace=True)]
            in_channels = i
    return layer


def add_extra(in_channels, stage):
    """
    only add CNN of brancdes S & L in stage Ti  at the end of net
    :param in_channels:the input channels & out
    :param stage:
    :return:list of layers
    """
    layers = []
    for i in stage:
        conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
    return layers
















base_net = make_layer(vgg_cfg)
vgg19 = nn.Sequential(*base_net)
vgg = torch.nn.ModuleList(base_net)
print(vgg19)
