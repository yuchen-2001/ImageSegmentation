import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel
from collections import OrderedDict

__all__ = ["DeepLabV3"]


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        # TODO Problem 2.1
        # ================================================================================ #
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        # TODO Problem 2.1
        # ================================================================================ #
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # TODO Problem 2.1
        # ================================================================================ #
        x = self.global_avg_pooling(x)
        x = self.conv(x)
        return self.relu(self.bn(x))


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        # TODO Problem 2.1
        # ================================================================================ #
        out_channels = 256
        # Creating ASPPConv modules with different dilation rates
        self.conv1 = ASPPConv(in_channels, out_channels, atrous_rates[0])
        self.conv2 = ASPPConv(in_channels, out_channels, atrous_rates[1])
        self.conv3 = ASPPConv(in_channels, out_channels, atrous_rates[2])
        # Creating ASPPPooling module
        self.pooling = ASPPPooling(in_channels, out_channels)
        # Combining the outputs
        self.project = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        # TODO Problem 2.1
        # ================================================================================ #
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.pooling(x)
        # Upsampling the pooled feature
        feat4 = F.interpolate(feat4, size=feat1.size()[2:], mode='bilinear', align_corners=False)
        # Concatenating all features
        x = torch.cat((feat1, feat2, feat3, feat4), dim=1)
        # Projecting and applying batch normalization
        x = self.project(x)
        x = self.bn(x)
        return self.relu(x)


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
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        result = OrderedDict()

        x = features["out"]
        x = self.classifier(x)

        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            aux = features["aux"]
            aux = self.aux_classifier(aux)
            aux = F.interpolate(aux, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = aux

        return result


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()
        # TODO Problem 2.2
        # The model should have the following 3 arguments
        #   in_channels: number of input channels
        #   num_classes: number of classes for prediction
        #   aspp_dilate: atrous_rates for ASPP
        #   
        # ================================================================================ #
        self.aspp = ASPP(in_channels, atrous_rates=aspp_dilate)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

        self._init_weight()

    def forward(self, feature):
        # TODO Problem 2.2
        # ================================================================================ #
        x = self.aspp(feature)
        global_features = self.global_avg_pooling(feature)
        global_features = self.conv1(global_features)
        global_features = self.relu1(self.bn1(global_features))
        global_features = F.interpolate(global_features, size=feature.size()[2:], mode='bilinear', align_corners=False)

        x += global_features
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
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.aspp = ASPP(in_channels + 48, atrous_rates=aspp_dilate)

        self.conv1 = nn.Conv2d(in_channels + 48, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

        self._init_weight()

    def forward(self, feature):
        # TODO Problem 2.2
        # ================================================================================ #
        low_level = 0###modify this
        x = self.aspp(feature)

        x = F.interpolate(x, size=low_level.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level], dim=1)

        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = F.interpolate(x, size=feature.size()[2:], mode='bilinear', align_corners=False)
        x = self.conv2(x)
        return x


def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
