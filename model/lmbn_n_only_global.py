import copy
import torch
from torch import nn
from .osnet import osnet_x1_0, OSBlock
from .attention import BatchDrop, BatchFeatureErase_Top, PAM_Module, CAM_Module, SE_Module, Dual_Module
from .bnneck import BNNeck, BNNeck3
from torch.nn import functional as F

from torch.autograd import Variable


class LMBN_n_only_global(nn.Module):
    def __init__(self, args):
        super(LMBN_n_only_global, self).__init__()

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

        self.global_branch = nn.Sequential(
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

        glo = self.global_branch(x)

        if self.batch_drop_block is not None:
            if self.activation_map:
                self.batch_drop_block.drop_batch_drop_top.training = True
            glo_drop, glo = self.batch_drop_block(glo)

        if self.activation_map:
            print('Generating activation maps...')
            return glo, glo_drop
        
        glo = self.channel_pooling(glo)  # shape:(batchsize, 512,1,1)
        glo_drop = self.global_pooling(glo_drop) # shape:(batchsize, 512,1,1)

        f_glo = self.reduction_0(glo)
        f_glo_drop = self.reduction_4(glo_drop)

        fea = [f_glo[-1], f_glo_drop[-1]]

        if not self.training:
            return torch.stack([f_glo[0], f_glo_drop[0]], dim=2)

        return [f_glo[1], f_glo_drop[1]], fea

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