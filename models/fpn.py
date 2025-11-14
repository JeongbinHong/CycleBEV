import math
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, reverse_outputs=True):
        """
        Args:
            in_channels_list (list): backbone 각 스테이지의 채널 수 [C2, C3, C4, C5]
            out_channels (int): FPN 출력 채널 수 (예: 256)
        """
        super(FPN, self).__init__()

        # # Lateral 1x1 conv layers to unify channel dimension
        # self.lateral_convs = nn.ModuleList([
        #     nn.Conv2d(in_ch, out_channels, kernel_size=1)
        #     for in_ch in in_channels_list
        # ])
        
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.GroupNorm(32, out_channels),  # or BatchNorm2d
                nn.ReLU(inplace=False),
            ) for in_ch in in_channels_list
        ])

        # # Output 3x3 conv layers for refinement
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
        self.reverse_outputs = reverse_outputs
        
    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): [C2, C3, C4, C5]
        Returns:
            list[Tensor]: [P2, P3, P4, P5]
        """
        assert len(inputs) == len(self.lateral_convs)

        # 1. Apply lateral 1x1 conv
        lateral_feats = [l_conv(x) for l_conv, x in zip(self.lateral_convs, inputs)]

        # 2. Top-down feature fusion
        num_levels = len(lateral_feats)
        pyramid_feats = [None] * num_levels
        pyramid_feats[-1] = lateral_feats[-1]  # P5

        for i in reversed(range(num_levels - 1)):
            upsample = F.interpolate(pyramid_feats[i+1], size=lateral_feats[i].shape[-2:], mode='bilinear', align_corners=False)
            pyramid_feats[i] = (lateral_feats[i] + upsample) / math.sqrt(2)

        # 3. Apply 3x3 conv for final output
        output_feats = [out_conv(p) for out_conv, p in zip(self.output_convs, pyramid_feats)]
        
        if self.reverse_outputs:
            return output_feats[::-1] # [P5, P4, P3, P2]
        else:
            return output_feats # [P2, P3, P4, P5]