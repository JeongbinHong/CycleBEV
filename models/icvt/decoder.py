import torch
import torch.nn as nn
import torch.nn.functional as F

    
class IVTDecoder(nn.Module):
    def __init__(self, in_channels=128, out_channels=3, hidden_dims=(64,32), target_size=(224, 480)):
        super(IVTDecoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.SiLU(inplace=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.SiLU(inplace=False),
        )
        self.conv_flatten = nn.Conv2d(hidden_dims[1], out_channels, kernel_size=1, bias=True)
        self.target_size = target_size


    def forward(self, x): 
        x = F.interpolate(x[0], scale_factor=2, mode='bilinear', align_corners=True) # [128, 56, 120]
        x = self.conv1(x)                                                            # [64, 56, 120]

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)    # [64, 112, 240] 
        x = self.conv2(x)                                                            # [32, 112, 240] 

        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)  # [32, H, W] 
        x = self.conv_flatten(x)                                                          # [C, H, W]
        return x


class IVTDecoder2(nn.Module):
    def __init__(self, in_channels=[128,128], out_channels=3, hidden_dims=(64,32), target_size=(224, 480)):
        super(IVTDecoder2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels[-1], hidden_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.SiLU(inplace=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-2], hidden_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.SiLU(inplace=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dims[0]*2, hidden_dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.SiLU(inplace=False),
        )
        self.conv_flatten = nn.Conv2d(hidden_dims[1], out_channels, kernel_size=1, bias=True)
        self.target_size = target_size


    def forward(self, features): # [128, 28, 60], [128, 28, 60]
        x1 = F.interpolate(features[-1], scale_factor=2, mode='bilinear', align_corners=True) # [128, 56, 120]
        x1 = self.conv1(x1)                                                                   # [64, 56, 120]

        x2 = F.interpolate(features[-2], scale_factor=2, mode='bilinear', align_corners=True) # [128, 56, 120] 
        x2 = self.conv2(x2)                                                                   # [64, 56, 120]

        x = torch.cat([x1, x2], dim=1)                                                        # [128, 56, 120]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)             # [128, 112, 240]
        x = self.conv3(x)                                                                     # [32, 112, 240]
        
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True) # [32, H, W] 
        x = self.conv_flatten(x)                                                         # [C,  H, W]
        return x
    
    
class IVTSelfSkip2Decoder(nn.Module):
    def __init__(self, in_channels=[128,128], out_channels=3, hidden_dims=(64,32), target_size=(224, 480)):
        super(IVTSelfSkip2Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels[-1], hidden_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.SiLU(inplace=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.SiLU(inplace=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-2], hidden_dims[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.SiLU(inplace=False),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_dims[1]+hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[1]),
            nn.SiLU(inplace=False),
        )
        self.conv_flatten = nn.Conv2d(hidden_dims[1], out_channels, kernel_size=1, bias=True)
        self.target_size = target_size


    def forward(self, features):
        x = F.interpolate(features[-1], scale_factor=2, mode='bilinear', align_corners=True)   # [128, 28, 60]
        x = self.conv1(x)                                                                       # [64, 28, 60]

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)              # [64, 56, 120] 
        x = self.conv2(x)                                                                       # [32, 56, 120] 

        y = F.interpolate(features[-2], scale_factor=2, mode='bilinear', align_corners=True)   # [128, 56, 120] 
        y = self.conv3(y)                                                                       # [64, 56, 120] 

        x = torch.cat([x, y], dim=1)                                                            # [96, 56, 120]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)              # [96, 112, 240]
        x = self.conv4(x)                                                                       # [32, 112, 240]

        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)       # [32, H, W] 
        x = self.conv_flatten(x)                                                                # [C, H, W]
        return x