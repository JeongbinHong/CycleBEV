import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
import torchvision.models as models
import copy


class CustomIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, drop_connect_rate=None):
        return x

class EffNet_Extractor(nn.Module):
    def __init__(self, num_classes, type='efficientnet-b4', layer_nums=[2,4], pretrained=True, channel_adapter=False):
        super(EffNet_Extractor, self).__init__()
        if pretrained:
            self.model = EfficientNet.from_pretrained(type)
        else:
            self.model = EfficientNet.from_name(type)
        #self.model.set_swish(memory_efficient=True)
        
        if channel_adapter:
            self.channel_adapter = nn.Sequential(nn.Conv2d(in_channels=num_classes, out_channels=3, kernel_size=1, bias=False),
                                                nn.BatchNorm2d(3),
                                                nn.SiLU())
        else:
            self.channel_adapter = None


        # reduction_5 : off (when b4)
        if type=='efficientnet-b4' and 5 not in layer_nums:
            for i in range(22, 32):
                self.model._blocks[i] = CustomIdentity()  # nn.Identity() 대신 CustomIdentity 사용

        # reduction_6 : off
        if 6 not in layer_nums:
            self.model._conv_head = nn.Identity()

        self.model._bn1 = nn.Identity()
        self.model._avg_pooling = nn.Identity()
        self.model._dropout = nn.Identity()
        self.model._fc = nn.Identity()
        self.model._swish = nn.Identity()

        self.layer_nums = layer_nums
        

    def forward(self, x):
        """
        (image size : 224 x 480)
        reduction_1 : [1, 24, 112, 240]
        reduction_2 : [1, 32, 56, 120]
        reduction_3 : [1, 56, 28, 60] 
        reduction_4 : [1, 160, 14, 30]  1/16
        reduction_5 : [1, 448, 7, 15]   1/32
        reduction_6 : [1, 1792, 7, 15]
        """
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        reductions = self.model.extract_endpoints(x)
        return [reductions[f'reduction_{i}'] for i in self.layer_nums]


class ResNet_Extractor(nn.Module):
    def __init__(self, num_classes, type='resnet50', layer_nums=[1,2], pretrained=True, channel_adapter=False):
        super(ResNet_Extractor, self).__init__()
        if type=='resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif type=='resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif type=='resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        self.layer_nums = layer_nums

        if channel_adapter:
            self.channel_adapter = nn.Sequential(nn.Conv2d(in_channels=num_classes, out_channels=3, kernel_size=1, bias=False),
                                                nn.BatchNorm2d(3),
                                                nn.SiLU()
                                                )
        else:
            self.channel_adapter = None

        self.layer0 = self.model.layer1
        self.layer1 = self.model.layer2
        self.layer2 = self.model.layer3
        if 3 in layer_nums:
            self.layer3 = self.model.layer4
        else:
            self.model.layer4 = nn.Identity()
            self.layer3 = None

        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()
    
    def forward(self, x):
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
            
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        feat0 = self.layer0(x)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1) # [1, 1024, 14, 30]
        if self.layer3 is not None:
            feat3 = self.layer3(feat2) # [1, 2048, 7, 15]
        else:
            feat3 = None

        '''
        (image size : 224 x 480)
        
        <ResNet18>
        feat0 : [1, 64, 56, 120]
        feat1 : [1, 128, 28, 60]
        feat2 : [1, 256, 14, 30]
        feat3 : [1, 512, 7, 15]

        <ResNet50, ResNet101>
        feat0 : [1, 256, 56, 120]
        feat1 : [1, 512, 28, 60]
        feat2 : [1, 1024, 14, 30]
        feat3 : [1, 2048, 7, 15]
        '''

        feats = [feat0, feat1, feat2, feat3]
        return [feats[i] for i in self.layer_nums]