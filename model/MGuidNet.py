import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .ResNet import ResNet50


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x

# side output module
class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class conv_upsample(nn.Module):
    def __init__(self, channel):
        super(conv_upsample, self).__init__()
        self.conv = BasicConv2d(channel, channel, 1)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = F.upsample(x, size=target.size()[2:], mode='bilinear', align_corners=True)
        return x


class conv_upsample1(nn.Module):
    def __init__(self, channel):
        super(conv_upsample1, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 1)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = self.conv1(F.upsample(x, size=target.size()[2:], mode='bilinear', align_corners=True))
        return x


class conv_upsample2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(conv_upsample2, self).__init__()
        self.conv2 = nn.Sequential(
            BasicConv2d(in_planes, out_planes,
                        kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation)
        )

    def forward(self, input):
        x, target = input
        if x.size()[2:] != target.size()[2:]:
            x = self.conv2(F.upsample(x, size=target.size()[2:], mode='bilinear', align_corners=True))
        return x

# SGM
class unet_pro(nn.Module):
    def __init__(self, channel):
        super(unet_pro, self).__init__()
        self.upsampleu = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv23 = conv_upsample(channel)
        self.conv24 = conv_upsample(channel)
        self.conv25 = conv_upsample(channel)
        self.conv26 = conv_upsample(channel)
        self.conv27 = conv_upsample(channel)
        self.conv28 = conv_upsample(channel)
        self.conv_upsampleu1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampleu2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampleu3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampleue1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampleue2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampleue3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_catu1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 3, padding=1)
        )
        self.conv_catu2 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_catu3 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_catue1 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_catue2 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_catue3 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

    def forward(self, x1, x2, x3, x4, e1, e2, e3, e4):
        x3 = torch.cat((x3, self.conv_upsampleu1(self.conv23(x4, x3))), 1)
        x3 = self.conv_catu1(x3)

        x2 = torch.cat((x2, self.conv_upsampleu2(self.conv24(x3, x2))), 1)
        x2 = self.conv_catu2(x2)

        x1 = torch.cat((x1, self.conv_upsampleu3(self.conv25(x2, x1))), 1)
        x1 = self.conv_catu3(x1)

        e3 = torch.cat((e3, self.conv_upsampleue1(self.conv26(e4, e3))), 1)
        e3 = self.conv_catue1(e3)

        e2 = torch.cat((e2, self.conv_upsampleue2(self.conv27(e3, e2))), 1)
        e2 = self.conv_catue2(e2)

        e1 = torch.cat((e1, self.conv_upsampleue3(self.conv28(e2, e1))), 1)
        e1 = self.conv_catue3(e1)

        return x1, x2, x3, x4, e1, e2, e3, e4

#CGM
class DenseFusion(nn.Module):
    def __init__(self, channel):
        super(DenseFusion, self).__init__()
        self.conv1 = conv_upsample2(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            conv_upsample2(channel, channel, kernel_size=3, padding=4, dilation=4)
        )
        self.conv3 = nn.Sequential(
            conv_upsample2(channel, channel, kernel_size=3, padding=8, dilation=8)
        )
        self.conv4 = conv_upsample2(channel, channel, kernel_size=3, padding=1)
        self.conv5 = nn.Sequential(
            conv_upsample2(channel, channel, kernel_size=3, padding=4, dilation=4)
        )
        self.conv6 = conv_upsample2(channel, channel, kernel_size=3, padding=1)
        self.conv7 = conv_upsample2(channel, channel, kernel_size=3, padding=1)
        self.conv8 = nn.Sequential(
            conv_upsample2(channel, channel, kernel_size=3, padding=4, dilation=4)
        )
        self.conv9 = nn.Sequential(
            conv_upsample2(channel, channel, kernel_size=3, padding=8, dilation=8)
        )
        self.conv10 = conv_upsample2(channel, channel, kernel_size=3, padding=1)
        self.conv11 = nn.Sequential(
            conv_upsample2(channel, channel, kernel_size=3, padding=4, dilation=4)
        )
        self.conv12 = conv_upsample2(channel, channel, kernel_size=3, padding=1)

        self.conv_f1 = nn.Sequential(
            BasicConv2d(5 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f2 = nn.Sequential(
            BasicConv2d(4 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f3 = nn.Sequential(
            BasicConv2d(3 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f4 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

        self.conv_f5 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f6 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f7 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f8 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

    def forward(self, x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4):
        x_sf1 = x_s1 + self.conv_f1(torch.cat((x_s1, x_e1,
                                               self.conv1([x_e2, x_s1]),
                                               self.conv2([x_e3, x_s1]),
                                               self.conv3([x_e4, x_s1])), 1))
        x_sf2 = x_s2 + self.conv_f2(torch.cat((x_s2, x_e2,
                                               self.conv4([x_e3, x_s2]),
                                               self.conv5([x_e4, x_s2])), 1))
        x_sf3 = x_s3 + self.conv_f3(torch.cat((x_s3, x_e3,
                                               self.conv6([x_e4, x_s3])), 1))
        x_sf4 = x_s4 + self.conv_f4(torch.cat((x_s4, x_e4), 1))

        x_ef1 = x_e1 + self.conv_f5(x_e1 * x_s1 *
                                    self.conv7([x_s2, x_e1]) *
                                    self.conv8([x_s3, x_e1]) *
                                    self.conv9([x_s4, x_e1]))
        x_ef2 = x_e2 + self.conv_f6(x_e2 * x_s2 *
                                    self.conv10([x_s3, x_e2]) *
                                    self.conv11([x_s4, x_e2]))
        x_ef3 = x_e3 + self.conv_f7(x_e3 * x_s3 *
                                    self.conv12([x_s4, x_e3]))
        x_ef4 = x_e4 + self.conv_f8(x_e4 * x_s4)

        return x_sf1, x_sf2, x_sf3, x_sf4, x_ef1, x_ef2, x_ef3, x_ef4

#FEU
class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.meanpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.meanpool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.meanpool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.conv2P = BasicConv2d(channel, channel, 3, padding=1)
        self.conv4P = BasicConv2d(channel, channel, 3, padding=1)
        self.conv8P = BasicConv2d(channel, channel, 3, padding=1)
        self.convP1 = BasicConv2d(channel, channel, 3, padding=1)
        self.convP2 = BasicConv2d(channel, channel, 3, padding=1)
        self.bnF = nn.BatchNorm2d(32)

    def forward(self, xF):
        xF = self.convP1(xF)
        xF_2 = self.meanpool2(xF)
        xF_4 = self.meanpool4(xF)
        xF_8 = self.meanpool8(xF)
        xF2 = self.conv2P(xF_2)
        xF4 = self.conv4P(xF_4)
        xF8 = self.conv8P(xF_8)
        x2 = F.upsample(xF2, size=xF.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.upsample(xF4, size=xF.size()[2:], mode='bilinear', align_corners=True)
        x8 = F.upsample(xF8, size=xF.size()[2:], mode='bilinear', align_corners=True)
        xF = xF + x2 + x4 + x8
        xF = self.bnF(xF)
        xF = self.convP2(xF)
        return xF

#AGM
class FPI(nn.Module):
    def __init__(self, channel):
        super(FPI, self).__init__()
        self.conv13 = conv_upsample2(channel, channel, kernel_size=3, padding=1)
        self.conv14 = nn.Sequential(
            conv_upsample2(channel, channel, kernel_size=3, padding=4, dilation=4)
        )
        self.conv15 = nn.Sequential(
            conv_upsample2(channel, channel, kernel_size=3, padding=8, dilation=8)
        )
        self.conv16 = conv_upsample2(channel, channel, kernel_size=3, padding=1)
        self.conv17 = nn.Sequential(
            conv_upsample2(channel, channel, kernel_size=3, padding=4, dilation=4)
        )
        self.conv18 = conv_upsample2(channel, channel, kernel_size=3, padding=1)
        self.conv19 = conv_upsample2(channel, channel, kernel_size=3, padding=1)
        self.conv20 = nn.Sequential(
            conv_upsample2(channel, channel, kernel_size=3, padding=4, dilation=4)
        )
        self.conv21 = conv_upsample2(channel, channel, kernel_size=3, padding=1)
        self.conv22 = conv_upsample2(channel, channel, kernel_size=3, padding=1)

        self.FAM1 = FAM(channel)
        self.FAM2 = FAM(channel)
        self.FAM3 = FAM(channel)
        self.FAM4 = FAM(channel)
        self.FAM5 = FAM(channel)
        self.FAM6 = FAM(channel)

        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, 1, 1)
        )

    def forward(self, s1, s2, s3, s4):
        s1 = s1 + self.FAM1(s1 +
                       self.conv13([s2, s1]) +
                       1.5 * self.conv14([s3, s1]) +
                       2 * self.conv15([s4, s1]))
        s2 = s2 + self.FAM2(s2 +
                       1.5 * self.conv16([s3, s2]) +
                       2 * self.conv17([s4, s2]))
        s3 = s3 + self.FAM3(s3 +
                       2 * self.conv18([s4, s3]))
        s1 = s1 + self.FAM4(s1 +
                       1.5 * self.conv19([s2, s1]) +
                       2 * self.conv20([s3, s1]))
        s2 = s2 + self.FAM5(s2 +
                       2 * self.conv21([s3, s2]))
        s1 = s1 + self.FAM6(s1 +
                       2 * self.conv22([s2, s1]))
        x = self.output(s1)
        return x

class final_fusion(nn.Module):
    def __init__(self, channel):
        super(final_fusion, self).__init__()
        self.finalconv1 = BasicConv2d(2, 2, 3, padding=1)
        self.finalconv2 = BasicConv2d(2, 1, 1)
    def forward(self, sfinal, efinal):
        sfinal = self.finalconv1(torch.cat((sfinal, efinal),1))
        sfinal = self.finalconv2(sfinal)
        return sfinal


class MGuid(nn.Module):
    def __init__(self, channel=32):
        super(MGuid, self).__init__()
        self.resnet = ResNet50()
        self.reduce_s1 = Reduction(256, channel)
        self.reduce_s2 = Reduction(512, channel)
        self.reduce_s3 = Reduction(1024, channel)
        self.reduce_s4 = Reduction(2048, channel)
        self.reduce_e1 = Reduction(256, channel)
        self.reduce_e2 = Reduction(512, channel)
        self.reduce_e3 = Reduction(1024, channel)
        self.reduce_e4 = Reduction(2048, channel)

        self.df1 = DenseFusion(channel)

        self.pro1 = unet_pro(channel)

        self.output_s = FPI(channel)
        self.output_e = FPI(channel)

        self.finalc = final_fusion(channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.initialize_weights()

    def forward(self, x):
        size = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        # feature extraction
        x_s1 = self.reduce_s1(x1)
        x_s2 = self.reduce_s2(x2)
        x_s3 = self.reduce_s3(x3)
        x_s4 = self.reduce_s4(x4)
        x_e1 = self.reduce_e1(x1)
        x_e2 = self.reduce_e2(x2)
        x_e3 = self.reduce_e3(x3)
        x_e4 = self.reduce_e4(x4)

        # feature refinement
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.pro1(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df1(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)

        # feature fusion
        pred_s = self.output_s(x_s1, x_s2, x_s3, x_s4)
        pred_e = self.output_e(x_e1, x_e2, x_e3, x_e4)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)
        pred_e = F.upsample(pred_e, size=size, mode='bilinear', align_corners=True)

        sfinal_s = self.finalc(pred_s, pred_e)

        return pred_s, pred_e, sfinal_s

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        self.resnet.load_state_dict(res50.state_dict(), False)
