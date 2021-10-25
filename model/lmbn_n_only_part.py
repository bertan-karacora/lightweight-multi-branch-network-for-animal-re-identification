import copy
import torch
from torch import nn
from .osnet import osnet_x1_0, OSBlock
from .attention import BatchDrop, BatchFeatureErase_Top, PAM_Module, CAM_Module, SE_Module, Dual_Module
from .bnneck import BNNeck, BNNeck3
from torch.nn import functional as F

from torch.autograd import Variable


class LMBN_n_only_part(nn.Module):
    def __init__(self, args):
        super(LMBN_n_only_part, self).__init__()

        self.n_ch = 2
        self.chs = 512 // self.n_ch

        osnet = osnet_x1_0(pretrained=True)

        self.backone = nn.Sequential(
            osnet.conv1,
            osnet.maxpool,
            osnet.conv2,
            osnet.conv3[0]
        )

        conv3 = osnet.conv3[1:]

        self.partial_branch = nn.Sequential(
            copy.deepcopy(conv3),
            copy.deepcopy(osnet.conv4),
            copy.deepcopy(osnet.conv5)
        )

        self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.partial_pooling = nn.AdaptiveAvgPool2d((2, 1))
        self.channel_pooling = nn.AdaptiveAvgPool2d((1, 1))

        reduction = BNNeck3(512, args.num_classes, args.feats, return_f=True)

        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)

        self.shared = nn.Sequential(
            nn.Conv2d(self.chs, args.feats, 1, bias=False),
            nn.BatchNorm2d(args.feats),
            nn.ReLU(True)
        )
        self.weights_init_kaiming(self.shared)

        self.reduction_ch_0 = BNNeck(args.feats, args.num_classes, return_f=True)
        self.reduction_ch_1 = BNNeck(args.feats, args.num_classes, return_f=True)

        self.batch_drop_block = BatchFeatureErase_Top(512, OSBlock)

        self.activation_map = args.activation_map

    def forward(self, x):
        x = self.backone(x)

        par = self.partial_branch(x)

        g_par = self.global_pooling(par)  # shape:(batchsize, 512,1,1)
        p_par = self.partial_pooling(par)  # shape:(batchsize, 512,2,1)

        p0 = p_par[:, :, 0:1, :]
        p1 = p_par[:, :, 1:2, :]

        f_p0 = self.reduction_1(g_par)
        f_p1 = self.reduction_2(p0)
        f_p2 = self.reduction_3(p1)

        fea = [f_p0[-1]]

        if not self.training:
            return torch.stack([f_p0[0], f_p1[0], f_p2[0]], dim=2)

        return [f_p0[1], f_p1[1], f_p2[1]], fea

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)