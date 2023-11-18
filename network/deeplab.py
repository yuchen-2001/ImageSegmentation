import torch
from torch import nn
from torch.nn import functional as F
from .utils import _SimpleSegmentationModel
import network.backbone.resnet
__all__ = ["DeepLabV3"]


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, rate):
        super(ASPPConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        size = x.shape[-2:]
        x = self.global_avg_pooling(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()

        modules = []
        # 1 - Conv2d has k1x1 s1
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU()))

        # 2-4 - Conv2d have k3x3 s1 with different rate
        for i in range(len(atrous_rates)):
            rate = atrous_rates[i]
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 5 - ASPP Pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        dropout_rate = 0.1
        # 7 - final conv2d with dropout
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        progress = []
        for conv in self.convs:
            progress.append(conv(x))
        # 6 - concat
        result = torch.cat(progress, dim=1)
        result = self.project(result)
        return result


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(DeepLabV3, self).__init__(backbone, classifier)
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        # take H and W
        shape = x.shape[-2:]
        # return an OrderedDict Tensor
        features = self.backbone(x)
        x = features["out"]
        x = self.classifier(x)
        # bilinear interpolate
        x = F.interpolate(x, size=shape, mode='bilinear', align_corners=False)

        # aux classifier is optional
        if self.aux_classifier != None:
            aux = features["aux"]
            aux = self.aux_classifier(aux)
            aux = F.interpolate(aux, size=shape, mode='bilinear', align_corners=False)
            x = aux

        return x


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()
        # The model should have the following 3 arguments
        #   in_channels: number of input channels
        #   num_classes: number of classes for prediction
        #   aspp_dilate: atrous_rates for ASPP
        #
        # ================================================================================ #
        channels = 256
        self.aspp = ASPP(in_channels, atrous_rates=aspp_dilate)
        # self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, num_classes, kernel_size=1)
        self._init_weight()

    def forward(self, feature):
        x = self.aspp(feature)
        # x = self.global_avg_pooling(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHeadV3Plus(nn.Module):
    # in_channels = 256, low_level_channels = 2048
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        # TODO Problem 2.2
        # The model should have the following 4 arguments
        #   in_channels: number of input channels
        #   low_level_channels: number of channels for project
        #   num_classes: number of classes for prediction
        #   aspp_dilate: atrous_rates for ASPP
        #
        # ================================================================================ #

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels + 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels + 48),
            nn.ReLU()
        )

        self.aspp = ASPP(in_channels=in_channels + 48, atrous_rates=aspp_dilate)

        self.conv1 = nn.Conv2d(2352, 1024, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()
        self.conv1extra = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.bn1extra = nn.BatchNorm2d(256)
        self.relu1extra = nn.ReLU()
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

        self._init_weight()


    def forward(self, feature):
        low_level_feature = self.project(feature)
        # print("Low-level feature shape:", low_level_feature.shape)
        x = torch.cat([self.aspp(low_level_feature), low_level_feature], dim=1)
        # print("Output of ASPP shape:", self.aspp(low_level_feature).shape)
        # print("Concatenated output shape:", x.shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.relu1extra(self.bn1extra(self.conv1extra(x)))
        x = self.conv2(x)
        # print("Final output shape:", x.shape)
        return x

    def _init_weight(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
