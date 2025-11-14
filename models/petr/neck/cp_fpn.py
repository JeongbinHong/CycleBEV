# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import copy
import warnings


def simple_xavier_init(module, gain=1, bias=0, distribution='normal'):
    """
    Simplified version of Xavier initialization using PyTorch.

    Args:
        module (nn.Module): PyTorch module whose parameters need initialization.
        gain (float): Scaling factor for the Xavier initialization. Default is 1.
        bias (float): Value to initialize the bias. Default is 0.
        distribution (str): Distribution for weight initialization ('normal' or 'uniform').

    Raises:
        AssertionError: If distribution is not 'normal' or 'uniform'.
    """
    assert distribution in ['uniform', 'normal'], \
        "Distribution must be 'uniform' or 'normal'."

    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        elif distribution == 'normal':
            nn.init.xavier_normal_(module.weight, gain=gain)

    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def apply_xavier_init(model, gain=1, bias=0, distribution='normal'):
    """
    Apply Xavier initialization to all submodules in a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to initialize.
        gain (float): Scaling factor for the Xavier initialization.
        bias (float): Value to initialize the bias.
        distribution (str): Distribution for weight initialization ('normal' or 'uniform').
    """
    for module in model.modules():
        simple_xavier_init(module, gain=gain, bias=bias, distribution=distribution)



####This FPN remove the unused parameters which can used with checkpoint (with_cp = True in Backbone)
class CPFPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CPFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            # l_conv = ConvModule(
            #     in_channels[i],
            #     out_channels,
            #     1,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
            #     act_cfg=act_cfg,
            #     inplace=False)
            
            l_conv = nn.Conv2d(in_channels=in_channels[i],
                                out_channels=out_channels,
                                kernel_size=1)

            self.lateral_convs.append(l_conv)
            if i == 0 :
                # fpn_conv = ConvModule(
                #     out_channels,
                #     out_channels,
                #     3,
                #     padding=1,
                #     conv_cfg=conv_cfg,
                #     norm_cfg=norm_cfg,
                #     act_cfg=act_cfg,
                #     inplace=False)
                
                fpn_conv = nn.Conv2d(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    padding=1)

                self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                # extra_fpn_conv = ConvModule(
                #     in_channels,
                #     out_channels,
                #     3,
                #     stride=2,
                #     padding=1,
                #     conv_cfg=conv_cfg,
                #     norm_cfg=norm_cfg,
                #     act_cfg=act_cfg,
                #     inplace=False)
                
                extra_fpn_conv = nn.Conv2d(in_channels=out_channels,
                                        out_channels=out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)
                
                self.fpn_convs.append(extra_fpn_conv)

        # mmcv/runner/BaseModule
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.init_cfg:
            # print_log(
            #     f'initialize {module_name} with init_cfg {self.init_cfg}',
            #     logger=logger_name)
            self.initialize(self, self.init_cfg)
            if isinstance(self.init_cfg, dict):
                # prevent the parameters of
                # the pre-trained model
                # from being overwritten by
                # the `init_weights`
                if self.init_cfg['type'] == 'Pretrained':
                    return

            # for m in self.children():
            #     if hasattr(m, 'init_weights'):
            #         m.init_weights()
            #         # users may overload the `init_weights`
            #         update_init_info(
            #             m,
            #             init_info=f'Initialized by '
            #             f'user-defined `init_weights`'
            #             f' in {m.__class__.__name__} ')

            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) if i==0 else laterals[i] for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
    
    # mmcv/cnn/utils/weight-init.py
    def _initialize(self, module, cfg, wholemodule=False):
        # func = build_from_cfg(cfg, INITIALIZERS)
        # # wholemodule flag is for override mode, there is no layer key in override
        # # and initializer will give init values for the whole module with the name
        # # in override.
        # func.wholemodule = wholemodule
        # func(module)

        if cfg['type']=='Xavier':
            apply_xavier_init(model=module, distribution=cfg['distribution'])

    # mmcv/cnn/utils/weight-init.py
    def _initialize_override(self, module, override, cfg):
        if not isinstance(override, (dict, list)):
            raise TypeError(f'override must be a dict or a list of dict, \
                    but got {type(override)}')

        override = [override] if isinstance(override, dict) else override

        for override_ in override:

            cp_override = copy.deepcopy(override_)
            name = cp_override.pop('name', None)
            if name is None:
                raise ValueError('`override` must contain the key "name",'
                                f'but got {cp_override}')
            # if override only has name key, it means use args in init_cfg
            if not cp_override:
                cp_override.update(cfg)
            # if override has name key and other args except type key, it will
            # raise error
            elif 'type' not in cp_override.keys():
                raise ValueError(
                    f'`override` need "type" key, but got {cp_override}')

            if hasattr(module, name):
                self._initialize(getattr(module, name), cp_override, wholemodule=True)
            else:
                raise RuntimeError(f'module did not have attribute {name}, '
                                f'but init_cfg is {cp_override}.')

    # mmcv/cnn/utils/weight-init.py
    def initialize(self, module, init_cfg):
        """Initialize a module.

        Args:
            module (``torch.nn.Module``): the module will be initialized.
            init_cfg (dict | list[dict]): initialization configuration dict to
                define initializer. OpenMMLab has implemented 6 initializers
                including ``Constant``, ``Xavier``, ``Normal``, ``Uniform``,
                ``Kaiming``, and ``Pretrained``.
        Example:
            >>> module = nn.Linear(2, 3, bias=True)
            >>> init_cfg = dict(type='Constant', layer='Linear', val =1 , bias =2)
            >>> initialize(module, init_cfg)

            >>> module = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1,2))
            >>> # define key ``'layer'`` for initializing layer with different
            >>> # configuration
            >>> init_cfg = [dict(type='Constant', layer='Conv1d', val=1),
                    dict(type='Constant', layer='Linear', val=2)]
            >>> initialize(module, init_cfg)

            >>> # define key``'override'`` to initialize some specific part in
            >>> # module
            >>> class FooNet(nn.Module):
            >>>     def __init__(self):
            >>>         super().__init__()
            >>>         self.feat = nn.Conv2d(3, 16, 3)
            >>>         self.reg = nn.Conv2d(16, 10, 3)
            >>>         self.cls = nn.Conv2d(16, 5, 3)
            >>> model = FooNet()
            >>> init_cfg = dict(type='Constant', val=1, bias=2, layer='Conv2d',
            >>>     override=dict(type='Constant', name='reg', val=3, bias=4))
            >>> initialize(model, init_cfg)

            >>> model = ResNet(depth=50)
            >>> # Initialize weights with the pretrained model.
            >>> init_cfg = dict(type='Pretrained',
                    checkpoint='torchvision://resnet50')
            >>> initialize(model, init_cfg)

            >>> # Initialize weights of a sub-module with the specific part of
            >>> # a pretrained model by using "prefix".
            >>> url = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/'\
            >>>     'retinanet_r50_fpn_1x_coco/'\
            >>>     'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
            >>> init_cfg = dict(type='Pretrained',
                    checkpoint=url, prefix='backbone.')
        """
        if not isinstance(init_cfg, (dict, list)):
            raise TypeError(f'init_cfg must be a dict or a list of dict, \
                    but got {type(init_cfg)}')

        if isinstance(init_cfg, dict):
            init_cfg = [init_cfg]

        for cfg in init_cfg:
            # should deeply copy the original config because cfg may be used by
            # other modules, e.g., one init_cfg shared by multiple bottleneck
            # blocks, the expected cfg will be changed after pop and will change
            # the initialization behavior of other modules
            cp_cfg = copy.deepcopy(cfg)
            override = cp_cfg.pop('override', None)
            self._initialize(module, cp_cfg)

            if override is not None:
                cp_cfg.pop('layer', None)
                self._initialize_override(module, override, cp_cfg)
            else:
                # All attributes in module have same initialization.
                pass


    def xavier_init(self, module, gain=1, bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)




