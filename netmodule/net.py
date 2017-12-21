'''
this function consist several stages of the networks
1.vgg19
2.stage_i
'''
import torch
import torch.nn as nn
from torch.autograd.variable import Variable

# openpose use first 10 layers of vgg19
vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 256,
           128]  # , 512, 512, 'M', 512, 512, 512, 512, 'M'
stage = [[3, 3, 3, 1, 1], [7, 7, 7, 7, 7, 1, 1]]
# branches_cfg[0] is PAF,branches_cfg[1] is heatmap
branches_cfg = [[[128, 128, 128, 512, 38], [128, 128, 128, 512, 19]],
                [[128, 128, 128, 128, 128, 128, 38], [128, 128, 128, 128, 128, 128, 19]]]

stage_time = 6  # openpose has 6 stages


class open_pose(nn.Module):
    """build the neuaral net

    """

    def __init__(self, phase, base, stage, branches_cfg, all_stage=6):
        super(open_pose, self).__init__()
        self.phase = phase

        # openpose network
        self.vgg = nn.ModuleList(base)

        conf_bra_list = []
        paf_bra_list = []

        # param for branch network
        in_channels = 128

        for i in range(all_stage):
            if i > 0:
                branches = branches_cfg[1]
                conv_sz = stage[1]
            else:
                branches = branches_cfg[0]
                conv_sz = stage[0]

            paf_bra_list.append(nn.Sequential(*add_extra(in_channels, branches[0], conv_sz)))
            conf_bra_list.append(nn.Sequential(*add_extra(in_channels, branches[1], conv_sz)))
            in_channels = 185

        # to list
        self.conf_bra = nn.ModuleList(conf_bra_list)
        self.paf_bra = nn.ModuleList(paf_bra_list)

    def forward(self, x, mask):
        masks = Variable(mask).cuda()  # .expand(predication.shape)
        out_0 = x
        # the base transform
        for k in range(len(self.vgg)):
            out_0 = self.vgg[k](out_0)

        # local name space
        name = locals()
        confs = []
        pafs = []
        outs = []

        length = len(self.conf_bra)
        for i in range(length):
            name['conf_%s' % (i + 1)] = self.conf_bra[i](name['out_%s' % i]) * masks
            name['paf_%s' % (i + 1)] = self.paf_bra[i](name['out_%s' % i]) * masks
            name['out_%s' % (i + 1)] = torch.cat([name['conf_%s' % (i + 1)], name['paf_%s' % (i + 1)], out_0], 1)
            confs.append('conf_%s' % (i + 1))
            pafs.append('paf_%s' % (i + 1))
            outs.append('out_%s' % (i + 1))
        for i in range(length):
            confs[i] = name.get(confs[i])
            pafs[i] = name.get(pafs[i])
            outs[i] = name.get(outs[i])

        return confs, pafs, outs

    def add_stages(self, i=128, all_stage=6):
        pass

        '''
        # two branches1
        conf1 = paf1 = out
        for i in range(len(self.conf_bra[0])):
            conf1 = self.conf_bra[0][i](conf1)
        for j in range(len(self.paf_bra[0])):
            paf1 = self.paf_bra[0][j](paf1)
        # concate
        out1 = torch.cat([conf1, paf1, out], 1)

        # two branches2
        conf2 = paf2 = out1
        for i in range(len(self.conf_bra[1])):
            conf2 = self.conf_bra[1][i](conf2)
        for j in range(len(self.paf_bra[1])):
            paf2 = self.paf_bra[1][j](paf2)
        # concate
        out2 = torch.cat([conf2, paf2, out], 1)

        # two branches 3
        conf3 = paf3 = out2
        for i in range(len(self.conf_bra[2])):
            conf3 = self.conf_bra[2][i](conf3)
        for j in range(len(self.paf_bra[2])):
            paf3 = self.paf_bra[2][j](paf3)
        # concate
        out3 = torch.cat([conf3, paf3, out], 1)

        # two branches 4
        conf4 = paf4 = out3
        for i in range(len(self.conf_bra[3])):
            conf4 = self.conf_bra[3][i](conf4)
        for j in range(len(self.paf_bra[3])):
            paf4 = self.paf_bra[3][j](paf4)
        # concate
        out4 = torch.cat([conf4, paf4, out], 1)

        # two branches 5
        conf5 = paf5 = out4
        for i in range(len(self.conf_bra[4])):
            conf5 = self.conf_bra[4][i](conf5)
        for j in range(len(self.paf_bra[4])):
            paf5 = self.paf_bra[4][j](paf5)
        # concate
        out5 = torch.cat([conf5, paf5, out], 1)

        # two branches 5
        conf6 = paf6 = out5
        for i in range(len(self.conf_bra[5])):
            conf6 = self.conf_bra[5][i](conf6)
        for j in range(len(self.paf_bra[5])):
            paf6 = self.paf_bra[5][j](paf6)
        # concate
        # out6 = torch.cat([conf6, paf6, out], 1)

        return conf1, paf1, conf2, paf2, conf3, paf3, conf4, paf4, conf5, paf5, conf6, paf6
        '''


# used for add two branches as well as adatp to ceratin stage
def add_extra(i, branches_cfg, stage):
    """
    only add CNN of brancdes S & L in stage Ti  at the end of net
    :param in_channels:the input channels & out
    :param stage: size of filter
    :param branches_cfg: channels of image
    :return:list of layers
    """
    in_channels = i
    layers = []
    for k in range(len(stage)):
        padding = stage[k] // 2
        conv2d = nn.Conv2d(in_channels, branches_cfg[k], kernel_size=stage[k], padding=padding)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = branches_cfg[k]
    return layers


# this function is used for build base net structure
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


def build_pose(phase):
    net = open_pose(phase, make_layer(vgg_cfg), stage, branches_cfg)
    return net


if __name__ == "__main__":
    base_net = make_layer(vgg_cfg)
    vgg19 = nn.Sequential(*base_net)
    vgg = torch.nn.ModuleList(base_net)
    print(vgg19)
