import torch
import torch.nn as nn
import torchvision.models as models
from Models.ResNet import ResNet50
from torch.nn import functional as F


#####################################       BasicConv2d          ########################################


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


#####################################       conv3x3          ########################################

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


#####################################       TransBasicBlock          ########################################

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out


#####################################       cross-modal information exchange (CIE) module          ##############################

class CIE(nn.Module):
    def __init__(self, in_channel):
        super(CIE, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size = 3, stride = 1, padding = 1)
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size = 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, f1, f2):

        f = f1.mul(f2)
        f = self.relu(self.bn(self.conv(f)))
        f = self.maxpool(self.upsample(f))

        f1 = f + f1
        f2 = f + f2
        f1 = self.maxpool(self.upsample(f1))
        f2 = self.maxpool(self.upsample(f2))

        f = f1 + f2

        return f, f1, f2


#####################################        multi-scale convolution (MC) module          ##############################

class MC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MC, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


#####################################        hierarchical feature fusion (HFF) module  in high-levels        ##############################


class HFF_high(nn.Module):
    def __init__(self, channel):
        super(HFF_high, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(2*channel, channel, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(2*channel, channel, kernel_size = 3, stride = 1, padding = 1)

        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x1, x2, x3):

        x1 = self.relu(self.bn1(self.conv1(self.upsample(x1))))
        x1 = self.maxpool(self.upsample(x1))

        x2 = torch.cat((x1, x2), 1)
        x2 = self.relu(self.bn2(self.conv2(x2)))
        x2 = x2.mul(x1)
        x2 = self.upsample(self.maxpool(self.upsample(x2)))

        x3 = torch.cat((x2, x3), 1)
        x3 = self.relu(self.bn3(self.conv3(x3)))
        x3 = x3.mul(x2)
        x3 = self.maxpool(self.upsample(x3))

        return x3

#####################################       hierarchical feature fusion (HFF) module in low-levels         ##############################

class HFF_low(nn.Module):
    def __init__(self, channel):
        super(HFF_low, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(2*channel, channel, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(2*channel, channel, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(channel, 3*channel, kernel_size = 1, stride = 1, padding = 0)

        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x1, x2, x3):

        x1 = self.relu(self.bn1(self.conv1(self.upsample(x1))))
        x1 = self.maxpool(self.upsample(x1))

        x2 = torch.cat((x1, x2), 1)
        x2 = self.relu(self.bn2(self.conv2(x2)))
        x2 = x2.mul(x1)
        x2 = self.maxpool(self.upsample(x2))

        x3 = torch.cat((x2, x3), 1)
        x3 = self.relu(self.bn3(self.conv3(x3)))
        x3 = x3.mul(x2)
        x3 = self.maxpool(self.upsample(x3))
        x3 = self.conv4(x3)

        return x3


#####################################       reverse guidance mechanism      ################################

class Refine(nn.Module):
    def __init__(self):
        super(Refine,self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(32, 64, kernel_size = 1, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(32, 256, kernel_size = 1, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(32, 512, kernel_size = 1, stride = 1, padding = 0)


    def forward(self, attention, x1, x2, x3):

        attention1 = self.conv1(attention)
        attention2 = self.conv2(attention)
        attention3 = self.conv3(attention)

        x1 = x1 + torch.mul(x1, self.upsample2(attention1))
        x2 = x2 + torch.mul(x2, self.upsample2(attention2))
        x3 = x3 + torch.mul(x3, attention3)

        return x1, x2, x3
    
############################    Cross-modal hierarchical interaction network (HINet)    ###############################

class HINet(nn.Module):
    def __init__(self, channel=32):
        super(HINet, self).__init__()
        
        ######    Backbone model
        self.resnet_rgb = ResNet50('rgb')
        self.resnet_depth = ResNet50('rgbd')

        ######    Cross-modal Information Exchange (CIE) module
        self.cie0 = CIE(64)
        self.cie1 = CIE(256)
        self.cie2 = CIE(512)
        self.cie3 = CIE(1024)
        self.cie4 = CIE(2048)

        ######    Multi-scale Convolution (MC) module
        self.rfb4_1 = MC(2048, channel)
        self.rfb3_1 = MC(1024, channel)
        self.rfb2_1 = MC(512, channel)

        self.rfb2_2 = MC(512, channel)
        self.rfb1_2 = MC(256, channel)
        self.rfb0_2 = MC(64, channel)

        ######    hierarchical feature fusion (HFF)
        self.agg1 = HFF_high(channel)
        self.agg2 = HFF_low(channel)

        ######    Reverse Guidance Mechanism
        self.HA = Refine()

        ######    upsample function
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.inplanes = 32*2
        self.deconv1 = self._make_transpose(TransBasicBlock, 32*2, 3, stride=2)
        self.inplanes =32
        self.deconv2 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.agant1 = self._make_agant_layer(32*3, 32*2)
        self.agant2 = self._make_agant_layer(32*2, 32)
        self.out0_conv = nn.Conv2d(32*3, 1, kernel_size=1, stride=1, bias=True)
        self.out1_conv = nn.Conv2d(32*2, 1, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(32*1, 1, kernel_size=1, stride=1, bias=True)

        if self.training:
            self.initialize_weights()

    def forward(self, x_rgb, x_depth):

        x_rgb = self.resnet_rgb.conv1(x_rgb)
        x_rgb = self.resnet_rgb.bn1(x_rgb)
        x_rgb = self.resnet_rgb.relu(x_rgb)
        x_rgb = self.resnet_rgb.maxpool(x_rgb)

        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)
        x_depth = self.resnet_depth.maxpool(x_depth)

        temp0, temp_rgb, temp_depth = self.cie0(x_rgb, x_depth)
        x_rgb_0 = x_rgb + temp_rgb
        x_depth_0 = x_depth + temp_depth


        x_rgb_1 = self.resnet_rgb.layer1(x_rgb_0)
        x_depth_1 = self.resnet_depth.layer1(x_depth_0)

        temp1, temp_rgb_1 ,temp_depth_1 = self.cie1(x_rgb_1, x_depth_1)
        x_rgb_1 = x_rgb_1 + temp_rgb_1
        x_depth_1 = x_depth_1 + temp_depth_1


        x_rgb_2 = self.resnet_rgb.layer2(x_rgb_1)
        x_depth_2 = self.resnet_depth.layer2(x_depth_1)

        temp2, temp_rgb_2 ,temp_depth_2 = self.cie2(x_rgb_2, x_depth_2)
        x_rgb_2 = x_rgb_2 + temp_rgb_2
        x_depth_2 = x_depth_2 + temp_depth_2


        x_rgb_3_1 = self.resnet_rgb.layer3_1(x_rgb_2)
        x_depth_3_1 = self.resnet_depth.layer3_1(x_depth_2)

        temp3, temp_rgb_3_1 ,temp_depth_3_1 = self.cie3(x_rgb_3_1, x_depth_3_1)
        x_rgb_3_1 = x_rgb_3_1 + temp_rgb_3_1
        x_depth_3_1 = x_depth_3_1 + temp_depth_3_1


        x_rgb_4_1 = self.resnet_rgb.layer4_1(x_rgb_3_1)
        x_depth_4_1 = self.resnet_depth.layer4_1(x_depth_3_1)

        temp4, temp_rgb_4_1 ,temp_depth_4_1 = self.cie4(x_rgb_4_1, x_depth_4_1)


        x4_1 = temp4
        x3_1 = temp3
        x2_1 = temp2
        x2 = temp2
        x1 = temp1
        x0 = temp0

        x2_1 = self.rfb2_1(x2_1)
        x3_1 = self.rfb3_1(x3_1)
        x4_1 = self.rfb4_1(x4_1)
        attention_map = self.agg1(x4_1, x3_1, x2_1)

        x0, x1, x2 = self.HA(attention_map.sigmoid(), x0, x1, x2)

        x2 = self.rfb2_2(x2)
        x1 = self.rfb1_2(x1)
        x0 = self.rfb0_2(x0)

        y = self.agg2(x2, x1, x0)

        y =self.agant1(y)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        y = self.out2_conv(y)
        attention_map = self.out2_conv(attention_map)

        return self.upsample(attention_map), y


# --------------------------------------------------------

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
    
    #initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet_rgb.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_rgb.state_dict().keys())
        self.resnet_rgb.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k=='conv1.weight':
                all_params[k]=torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)



