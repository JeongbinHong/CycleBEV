import os.path
import math
from utils.functions import *
from utils.loss import *
from utils.metrics import IoUMetric
from models.icvt.icvt import InverseCrossViewTransformer
from torch.nn.parallel import DistributedDataParallel as DDP
from models.extractors import EffNet_Extractor, ResNet_Extractor
from models.icvt.decoder import *
import torch.distributed as dist
from utils.visualizer import save_images
from einops import rearrange
from utils.functions import Normalize
from fvcore.nn import sigmoid_focal_loss
from utils.loss import Optimizers
import torch.optim as optim
import copy
from utils.visualizer  import BaseViz

class FullModel(nn.Module):
    def __init__(self, cfg, args, backbone, vt, ivt, decoder, norm, num_batches):
        super().__init__()
        self.cfg = cfg
        self.args = args
        if backbone is not None:
            self.backbone = backbone
        self.vt = vt
        self.ivt = ivt
        self.decoder = decoder
        self.norm = norm
        threshold = float(args.iou_threshold.replace("@",""))
        self.offset = torch.log(torch.tensor(threshold / (1 - threshold))) # threshold=0.5 → offset=0.0
        
        self.noise_mode = cfg['IVT']['bev']['noise_mode']
        self.noise_std = cfg['IVT']['bev']['noise_std']
        self.noise_converging = cfg['IVT']['bev']['noise_converging']
        
        if self.args.w_feat_loss != 0:
            dim = cfg[f'{args.vt_model}']['decoder']['dim']
            outDim = cfg['IVT']['encoder']['fpn']['out_channels']
            self.projector = nn.Conv2d(dim, outDim, kernel_size=1)
        
    def forward(self, batch, isTrain=True):
        if self.args.vt_model in {'LSS', 'PETR'}:
            bev_pred, vt_output = self.vt(batch, isTrain)
        elif self.args.vt_model == 'CVT':
            img_features = self.backbone(self.norm(batch['image']))
            bev_pred, vt_output = self.vt(batch, img_features, isTrain)
        elif self.args.vt_model == 'BEVFormer':
            input_images = self.norm(batch['image'])
            if len(input_images.shape) > 4:
                img_features = [self.backbone(img) for img in input_images]
            else:
                img_features = self.backbone(input_images)
            bev_pred, vt_output = self.vt(batch, img_features, isTrain)
        else:
            raise ValueError(f"Unsupported vt_model: {self.args.vt_model}")

        if isTrain:
            # Cycle Consistency Path : Image → BEV → PV
            if self.cfg['IVT']['logit_processing'] == 'sigmoid':
                pred_bev_for_ivt = {}
                for key, val in bev_pred.items():
                    if key != 'height':
                        pred_bev_for_ivt[key] = [torch.sigmoid(val[0] - self.offset)]
                    else:
                        pred_bev_for_ivt[key] = [val[0]] 
                bev_for_ivt = [pred_bev_for_ivt[key][0].squeeze(1) for key, value in pred_bev_for_ivt.items()]

            elif self.cfg['IVT']['logit_processing'] == 'raw':
                bev_for_ivt = [bev_pred[key][0].squeeze(1) for key, value in bev_pred.items()]

            bev_for_ivt = torch.stack(bev_for_ivt, dim=1)
            _batch = copy.deepcopy(batch)
            _batch['bev'] = bev_for_ivt
            ivt_outputs_cc, _ = self.ivt(_batch)
            pvcc_pred = self.decoder(ivt_outputs_cc)
        
            # Ground-truth Guide Path : BEV → PV
            if self.noise_mode != 'none':
                noise_bev = batch['bev'].clone() + torch.randn_like(batch['bev']) * self.noise_std
                
                if self.noise_mode == 'hard':
                    noise_bev = torch.clamp(noise_bev, 0, 1)

                elif self.noise_mode == 'soft':
                    noise_bev = noise_bev.abs()
                    excess = (noise_bev - 1).clamp(min=0)
                    noise_bev = noise_bev - 2 * excess

                _batch['bev'] = noise_bev
            else:
                _batch['bev'] = batch['bev']
                
            if self.args.get_height:
                _batch['bev'] = torch.concat([_batch['bev'], batch['bev_height']], dim=1)

            ivt_outputs_gg, bev_feature = self.ivt(_batch)
            pvgg_pred = self.decoder(ivt_outputs_gg)
            
            if self.args.w_feat_loss != 0.0:
                bev_feat = bev_feature.detach().clone()
                if bev_feat.shape[-2:] != vt_output.shape[-2:]:
                    vt_output = F.interpolate(vt_output, size=bev_feat.shape[-2:], mode='bilinear', align_corners=True)
                vt_output = self.projector(vt_output) # aply the projector in all cases.
            else:
                vt_output, bev_feat = None, None
                
            return bev_pred, pvcc_pred, pvgg_pred, (vt_output, bev_feat)

        else:
            return bev_pred
    


class Solver:

    def __init__(self, cfg, args, num_train_scenes, world_size=None, rank=None, logger=None, dtype=None, isTrain=True):
        seed_fixer(args.random_seed)
        self.save_dir = args.save_dir

        # load pre-trained settings or save current settings
        if (isTrain):
            if (args.load_pretrained == 1):
                if (os.path.exists(self.save_dir) is not True): sys.exit(f'>> path {self.save_dir} does not exist!!')
                else:
                    with open(os.path.join(self.save_dir, 'config.pkl'), 'rb') as f:
                        args = pickle.load(f)

            # or save current settings
            else:
                if (rank==0):
                    with open(os.path.join(self.save_dir, 'config.pkl'), 'wb') as f:
                        pickle.dump(args, f)

        # training setting
        self.args = args
        self.cfg = cfg # 20250904
        self.rank, self.world_size = rank, world_size

        self.log = logger
        
        self.dtype = dtype
        if args.drop_last:
            self.num_batches = int(num_train_scenes / (args.batch_size * world_size))
        else:
            self.num_batches = math.ceil(num_train_scenes / (args.batch_size * world_size))

        # training monitoring
        self.monitor = {'iter': 0,
                        'loss_sum':0,
                        'bev_loss_sum':0,
                        'pvcc_loss_sum':0,
                        'pvgg_loss_sum':0,
                        'feat_loss_sum':0,
                        'val_miou': 0,
                        'val_miou_vis0': 0,
                        'val_miou_vis40up': 0,
                        'val_iou_dri_vis0': 0,
                        'val_iou_veh_vis0': 0,
                        'val_iou_veh_vis40up': 0,
                        'val_iou_ped_vis0': 0,
                        'val_iou_ped_vis40up': 0,
                        'prev_miou': 0
                        # 'cur_lr': cfg['learning_rate']
                        }
        
        self.save_target_output = []
        

        self.norm = Normalize('imagenet')
        self.norm_ns = Normalize('nuscenes_topcrop')
        
        img_feat_shapes = []
        if args.vt_model == 'PETR':
            backbone = None
        else:
            backbone_name = self.cfg[args.vt_model]['backbone']['model_name']
            layer_nums = self.cfg[args.vt_model]['backbone']['layer_nums']  
            if 'resnet' in backbone_name:
                backbone = ResNet_Extractor(num_classes=3, type=backbone_name, layer_nums=layer_nums)
            elif 'efficientnet' in backbone_name:
                backbone = EffNet_Extractor(num_classes=3, type=backbone_name, layer_nums=layer_nums)
            
            with torch.no_grad():
                backbone = backbone.to(rank)
                sample = torch.randn(1, 3, args.img_h, args.img_w, device=rank)
                features = backbone(sample)
                img_feat_shapes = [feature.shape for feature in features]
            
        if args.vt_model == 'CVT':
            from models.cvt.cvt import CrossViewTransformer
            vt = CrossViewTransformer(self.cfg, self.args, img_feat_shapes)
            
        elif args.vt_model == 'BEVFormer':
            if self.cfg['use_temporal']:
                from models.bevformer.bevformer_temp import BEVformer
            else:
                from models.bevformer.bevformer import BEVformer
            vt = BEVformer(self.cfg, self.args, img_feat_shapes)
            
        elif args.vt_model == 'LSS':
            from models.lss.lss import LiftSplatShoot
            vt = LiftSplatShoot(self.cfg, self.args, img_feat_shapes)
            backbone = None
            
        elif args.vt_model == 'PETR':
            from models.petr.petr import PETR
            vt = PETR(self.cfg, self.args)
            backbone = None
            
        ivt_feat_shapes = self.cfg['IVT']['encoder']['ivt_feat_shapes']
        bev_layer_nums = self.cfg['IVT']['encoder']['bev_layer_nums']
        ivt_decoder_type = self.cfg['IVT']['decoder']['type']
        ivt_train_mode = self.cfg['IVT']['ivt_train_mode']
        pretrained_target_num = self.cfg['IVT']['pretrained_target_num']
        final_mask_h, final_mask_w = self.cfg['IVT']['image']['final_mask_h'], self.cfg['IVT']['image']['final_mask_w']
        
        if ivt_train_mode == 'pretrain2finetuning':
            from_pretrain = True
            out_channels = pretrained_target_num + 1 # with background
        elif ivt_train_mode == 'pretrain2freeze':
            from_pretrain = True
            out_channels = len(args.targets) + 1 # with background
        else:
            from_pretrain = False
            out_channels = len(args.targets) + 1 # with background

        in_channels = [shapes[1] for shapes in ivt_feat_shapes]
        out_channels = len(args.targets)+1 # with background
        target_size = (final_mask_h, final_mask_w)
        if ivt_decoder_type=='IVTDecoder':
            decoder = IVTDecoder(in_channels[0], out_channels, target_size=target_size)
        elif ivt_decoder_type=='IVTDecoder2':
            decoder = IVTDecoder2(in_channels, out_channels, target_size=target_size)
        elif ivt_decoder_type=='IVTSelfSkip2Decoder':
            decoder = IVTSelfSkip2Decoder(in_channels, out_channels, target_size=target_size)
        else:
            raise ValueError(f"'{ivt_decoder_type}' is not a supported decoder.")
        
        ivt = InverseCrossViewTransformer(self.cfg, self.args, ivt_feat_shapes, bev_layer_nums, rank=rank, from_pretrain=from_pretrain)
        if 'pretrain' in ivt_train_mode:
            ivt = self.load_pretrained_model(ivt, args.pretrained_ivt_path, 'ivt')
            decoder = self.load_pretrained_model(decoder, args.pretrained_ivt_path, 'decoder')

            if ivt_train_mode == 'pretrain2freeze':
                for param in ivt.parameters():
                    param.requires_grad = False  
                for param in decoder.parameters():
                    param.requires_grad = False 
                
        if self.args.get_height == 0:
            in_channels = 3
            ivt.encoder.ivt_backbone.channel_adapter = nn.Sequential(nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
                                                                nn.BatchNorm2d(3),
                                                                nn.SiLU())   
            
        if self.args.load_trained_vt:
            if args.vt_model in {'LSS', 'PETR'}:
                backbone = None
            elif args.vt_model in {'CVT', 'BEVFormer'}:
                backbone = self.load_pretrained_model(backbone, args.trained_vt_path, 'backbone')
            vt = self.load_pretrained_model(vt, args.trained_vt_path, 'vt')                                                          
            
        model = FullModel(self.cfg, self.args, backbone, vt, ivt, decoder, self.norm, self.num_batches)

        # DDP
        if args.ddp: 
            torch.cuda.set_device(rank)
            model = model.type(dtype).to(rank)
            self.model = DDP(model, device_ids=[rank], find_unused_parameters=bool(self.args.bool_find_unused_params))
            m = self.model.module
        else:
            self.model = model.type(dtype).cuda()
            m = self.model

        # define optimizer
        # self.opt = Optimizers(self.model, optimizer_type=args.optimizer_type,
        #                      learning_rate=args.learning_rate, weight_decay=args.weight_decay).opt
        if self.args.vt_model in {'LSS', 'PETR'}:
            vt_params = [*m.vt.parameters()]
        elif self.args.vt_model in {'CVT', 'BEVFormer'}:
            vt_params = [*m.backbone.parameters(), *m.vt.parameters()]
        else:
            raise ValueError(f"Unsupported vt_model: {self.args.vt_model}")
        
        ivt_params = [*m.ivt.parameters(), *m.decoder.parameters()]
        if self.args.w_feat_loss != 0.0:
            ivt_params += list(m.projector.parameters())
        
        train_set = self.cfg[f'{args.vt_model}']['training']
        num_epochs = train_set['num_epochs']
        learning_rate = train_set['learning_rate']
        weight_decay = train_set['weight_decay']
        div_factor = train_set['div_factor']
        pct_start = train_set['pct_start']
        final_div_factor = train_set['final_div_factor']
        
        train_set_ivt = self.cfg[f'IVT']['training']
        learning_rate2 = train_set_ivt['learning_rate']
        weight_decay2 = train_set_ivt['weight_decay']
        div_factor2 = train_set_ivt['div_factor']
        pct_start2 = train_set_ivt['pct_start']
        final_div_factor2 = train_set_ivt['final_div_factor']
            
        param_groups = [
            {'params': vt_params,
            'lr': learning_rate,
            'weight_decay': weight_decay}
        ]
        
        param_groups2 = [
            {'params': ivt_params,
            'lr': learning_rate2,
            'weight_decay': weight_decay2}
            ]
            
        self.opt = optim.AdamW(param_groups)
        self.opt2 = optim.AdamW(param_groups2)

        # training schedule
        if self.args.lr_schd_type != 'none':
            lr_cfg = {'max_lr':learning_rate, 
                        'div_factor': div_factor,
                        'final_div_factor': final_div_factor, 'pct_start':pct_start,
                        'steps_per_epoch': self.num_batches, 'epochs':num_epochs,  
                        }
            self.lr_scheduler = LRScheduler(self.opt, type=self.args.lr_schd_type, config=lr_cfg)
            
            lr_cfg2 = {'max_lr':learning_rate2, 
                        'div_factor': div_factor2,
                        'final_div_factor': final_div_factor2, 'pct_start':pct_start2,
                        'steps_per_epoch': self.num_batches, 'epochs':num_epochs,  
                        }
            self.lr_scheduler2 = LRScheduler(self.opt2, type=self.args.lr_schd_type, config=lr_cfg2)

        if len(self.args.targets)==3:
            assert self.args.targets == ['drivable', 'vehicle', 'pedestrian']

        self.class_id = {'drivable': 1,
                         'vehicle': 2,
                         'pedestrian': 3}
        self.target_nums = [self.class_id[target] for target in self.args.targets]
        
        self.class_weights_vt = {'drivable': args.w_vt_dri,
                                'vehicle': args.w_vt_veh,
                                'pedestrian': args.w_vt_ped,
                                'height': args.w_vt_height
                                }
        self.class_weights_ivt = {'drivable': args.w_ivt_dri,
                                'vehicle': args.w_ivt_veh,
                                'pedestrian': args.w_ivt_ped,
                                }

        self.alpha = [self.cfg['Loss']['target_focal'][target]['alpha'] for target in self.args.targets]
        self.gamma = [self.cfg['Loss']['target_focal'][target]['gamma'] for target in self.args.targets]
            
        min_visibility = None
        if args.visibility and ('vehicle' in self.args.targets or 'pedestrian' in self.args.targets):
            min_visibility = 2
        self.bev_loss = LossScratch(cfg=self.cfg, args=self.args, min_visibility=min_visibility)
        
        if (rank == 0):
            print(">> Optimizer is loaded from {%s} " % os.path.basename(__file__))

        Thresholds = {'drivable': 0.3,
                      'vehicle': 0.1,
                      'pedestrian': 0.0,
                        'lane': 0.4,
                        'road': 0.4,
                        }
        self.vis = BaseViz(cfg=self.cfg, 
                           args=self.args,
                           targets=self.args.targets, 
                            label_indices=self.cfg['label_indices'], 
                            Thresholds=Thresholds)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def mode_selection(self, isTrain=True):
        if (isTrain): 
            self.model.train()
        else: 
            self.model.eval()


    def init_loss_tracker(self):
        self.monitor['loss_sum'] = 0
        self.monitor['bev_loss_sum'] = 0
        self.monitor['pvcc_loss_sum'] = 0
        self.monitor['pvgg_loss_sum'] = 0
        self.monitor['feat_loss_sum'] = 0
        self.monitor['max_grad'] = 0

    def normalize_loss_tracker(self):
        self.monitor['loss_sum'] /= self.num_batches
        self.monitor['bev_loss_sum'] /= self.num_batches
        self.monitor['pvcc_loss_sum'] /= self.num_batches
        self.monitor['pvgg_loss_sum'] /= self.num_batches
        self.monitor['feat_loss_sum'] /= self.num_batches

    def lr_scheduler_step(self):
        if self.args.lr_schd_type != 'none':
            self.lr_scheduler()
            self.lr_scheduler2()

    def load_pretrained_model(self, model, path, name=None):
        ckp_idx = save_read_latest_checkpoint_num(path, 0, isSave=False)

        if name != None:
            file_name = path + f'/saved_chk_point_{name}_{ckp_idx}.pt'
        else:
            file_name = path + f'/saved_chk_point_{ckp_idx}.pt'
        checkpoint = torch.load(file_name, map_location=torch.device('cpu'))

        if self.args.ddp:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k.replace("module.", "") # removing ‘module.’ from key
                new_state_dict[name] = v

            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model.state_dict()}
            model.load_state_dict(pretrained_dict, strict=True)

        if self.rank == 0:
            self.log.info('>> trained parameters are loaded from {%s}' % file_name)
            self.log.info(">> current training status : %.4f mIoU" % checkpoint['prev_miou'])

        return model


    def save_trained_network_params(self, model, name, e):
        # save trained model
        _ = save_read_latest_checkpoint_num(os.path.join(self.save_dir), e, isSave=True)
        if name != None:
            file_name = self.save_dir + f'/saved_chk_point_{name}_{e}.pt'
        else:
            file_name = self.save_dir + f'/saved_chk_point_{e}.pt'

        check_point = {
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'opt': self.opt.state_dict(),
            'prev_miou': self.monitor['prev_miou'],
            'iter': self.monitor['iter'],
            'cfg': self.cfg}
    
        torch.save(check_point, file_name)
        self.log.info(f">> 💾 current network is saved.")
        remove_past_checkpoint(os.path.join('./', self.save_dir), self.args.remain_num_chkpts, name)

    def print_status(self, e, tl): # average loss for each epoch
        if (self.rank==0):
            cur_lr = self.opt.param_groups[0]['lr']
            cur_lr2 = self.opt2.param_groups[0]['lr']
            self.log.info(f"[Epoch {e:d}, {tl:.2f} hrs left] "
                          f"loss: {self.monitor['loss_sum']:.4f}, "
                          f"bev_loss: {self.monitor['bev_loss_sum']:.4f}, "
                          f"pvcc_loss: {self.monitor['pvcc_loss_sum']:.4f}, "
                          f"pvgg_loss: {self.monitor['pvgg_loss_sum']:.4f}, "
                          f"feat_loss: {self.monitor['feat_loss_sum']:.4f}, "
                          f"(cur lr: {cur_lr:.7f}, {cur_lr2:.7f})")


    def print_training_progress(self, e, b, time): # real time loss
        if (self.rank == 0):
            if (b >= self.num_batches - 2): sys.stdout.write('\r')
            else: sys.stdout.write(f"\r [Epoch {e}] {b} / {self.num_batches} ({time:.4f} sec/sample), " 
                                   f"loss: {self.monitor['loss']:.4f}, "
                                   f"bev_loss: {self.monitor['bev_loss']:.4f}, "
                                   f"pvcc_loss: {self.monitor['pvcc_loss']:.4f}, "
                                   f"pvgg_loss: {self.monitor['pvgg_loss']:.4f}, "
                                   f"feat_loss: {self.monitor['feat_loss']:.4f}, "
                                   #f" lr: {self.opt.param_groups[0]['lr']:.7f}"
                                   ),
            sys.stdout.flush()

    def print_validation_progress(self, b, num_batchs):
        if (self.rank == 0):
            if (b >= num_batchs - 2): sys.stdout.write('\r')
            else: sys.stdout.write('\r >> validation process (%d / %d) ' % (b, num_batchs)),
            sys.stdout.flush()

    def wandb_tracker(self,):
        CFG = {
            'cfg' : self.cfg,
            'train_loss' : self.monitor['loss_sum'],
            'val_miou' : self.monitor['val_miou'],
            'val_miou_vis0' : self.monitor['val_miou_vis0'],
            'val_miou_vis40' : self.monitor['val_miou_vis40up'],
            'val_iou_dri_vis0' : self.monitor['val_iou_dri_vis0'],
            'val_iou_veh_vis0' : self.monitor['val_iou_veh_vis0'],
            'val_iou_veh_vis40' : self.monitor['val_iou_veh_vis40up'],
            'val_iou_ped_vis0' : self.monitor['val_iou_ped_vis0'],
            'val_iou_ped_vis40' : self.monitor['val_iou_ped_vis40up']
        }
        return CFG

    # ------------------------
    # Training
    # ------------------------
    def reform_batch(self, batch):
        for key, value in batch.items():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].type(self.dtype).to(self.rank)
                if self.args.vt_model in {'BEVFormer' 'PETR'}:
                    if key not in {'image','intrinsics','extrinsics'}:
                        if len(batch[key].shape)>=5 and batch[key].size(1)==1:
                            batch[key] = batch[key][:, 0]
                else:
                    if len(batch[key].shape)>=5 and batch[key].size(1)==1:
                        batch[key] = batch[key][:, 0] # remove temporal dimension

            if key == 'image':
                if self.cfg['use_temporal']:
                    if self.args.vt_model=='BEVFormer':
                        batch[key] = rearrange(batch[key], 'b t n c h w -> t (b n) c h w')
                else:
                    batch[key] = rearrange(batch[key], 'b n c h w -> (b n) c h w')
            elif key in {'intrinsics','extrinsics'}:
                if len(batch[key].shape)==5 :
                    if self.args.vt_model == 'BEVFormer':
                        batch[key] = rearrange(batch[key], 'b t n h w -> t b n h w')
            elif key in {'image_seg_masks'}:
                batch[key] = rearrange(batch[key], 'b n h w -> (b n) h w')
        return batch


    def train(self, batch, e, b):  # train.py - solver.train(data)  ->  batch = data
        batch = self.reform_batch(batch)
        target_mask = batch['image_seg_masks']
        target_mask = F.one_hot(target_mask.long(), num_classes=4)
        target_mask = target_mask.permute(0, 3, 1, 2).float()

        self.opt.zero_grad()
        self.opt2.zero_grad()
        bev_pred, pvcc_pred, pvgg_pred, (vt_output, bev_feature) = self.model(batch, isTrain=True)  #
        '''
        pred_bev : dict("class_name":list(tensor([b, 1, h, w])))
        pred_mask : tensor([b, num_classes+1, h, w])
        '''
        bev_loss = torch.zeros(1).to(self.rank)
        bev_loss_dict = self.bev_loss.main(bev_pred, batch)
        for key, item in bev_loss_dict.items():
            bev_loss += self.class_weights_vt[key] * item['loss']
        
        target_mask = target_mask[:,self.target_nums]
        
        pvcc_pred = pvcc_pred[:,1:] # without background
        pvcc_loss_dict = [sigmoid_focal_loss(pvcc_pred[:,i], target_mask[:,i], self.alpha[i], self.gamma[i]) for i in range(pvcc_pred.size(1))]
        pvcc_loss_dict = torch.stack(pvcc_loss_dict, dim=1)
        pvcc_loss_dict = self.get_dict_loss(pvcc_loss_dict) # dict("class_name":list(tensor([b, 1, h, w])))

        pvgg_pred = pvgg_pred[:,1:]
        pvgg_loss_dict = [sigmoid_focal_loss(pvgg_pred[:,i], target_mask[:,i], self.alpha[i], self.gamma[i]) for i in range(pvgg_pred.size(1))]
        pvgg_loss_dict = torch.stack(pvgg_loss_dict, dim=1)
        pvgg_loss_dict = self.get_dict_loss(pvgg_loss_dict) # dict("class_name":list(tensor([b, 1, h, w])))

        
        pvcc_loss = torch.zeros(1).to(self.rank)
        pvgg_loss = torch.zeros(1).to(self.rank)
        for key, item in pvcc_loss_dict.items():
            pvcc_loss += self.class_weights_ivt[key] * item['loss']
        for key, item in pvgg_loss_dict.items():
            pvgg_loss += self.class_weights_ivt[key] * item['loss']

        if self.args.w_feat_loss != 0.0:
            if self.args.feat_loss_type in {'L1', 'mae'}:
                feat_loss = F.l1_loss(vt_output, bev_feature)
            elif self.args.feat_loss_type in {'smooth_L1'}:
                feat_loss = F.smooth_l1_loss(vt_output, bev_feature)  
            elif self.args.feat_loss_type in {'L2', 'mse'}:
                feat_loss = F.mse_loss(vt_output, bev_feature)
            elif self.args.feat_loss_type == 'cosim':
                feat_loss = F.cosine_similarity(
                    vt_output.reshape(vt_output.size(0),-1), 
                    bev_feature.reshape(vt_output.size(0),-1), dim=-1).mean()
                feat_loss = -feat_loss + 1
        else:
            feat_loss = torch.zeros(1).to(self.rank)
        
        bev_loss *= self.args.w_bev_loss
        pvcc_loss *= self.args.w_pvcc_loss
        pvgg_loss *= self.args.w_pvgg_loss
        feat_loss *= self.args.w_feat_loss

        # weights of each loss
        loss = bev_loss + pvcc_loss + pvgg_loss + feat_loss

        # back-propagation
        loss.backward()
        if self.args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.opt.step()
        self.opt2.step()

        self.monitor['loss'] = loss.item()
        self.monitor['bev_loss'] = bev_loss.item()
        self.monitor['pvcc_loss'] = pvcc_loss.item()
        self.monitor['pvgg_loss'] = pvgg_loss.item()
        self.monitor['feat_loss'] = feat_loss.item()

        self.monitor['loss_sum'] += loss.item()
        self.monitor['bev_loss_sum'] += bev_loss.item()
        self.monitor['pvcc_loss_sum'] += pvcc_loss.item()
        self.monitor['pvgg_loss_sum'] += pvgg_loss.item()
        self.monitor['feat_loss_sum'] += feat_loss.item()

        # increase iteration number
        self.monitor['iter'] += 1

        #get_grad_norm(self.model, log=self.log, mode='mean')


    # ------------------------
    # Validation
    # ------------------------
    def eval(self, dataset, dataloader, sampler, e):
        d_len = len(dataloader)
            
        if self.args.ddp:
            rank = dist.get_rank()
        else: 
            rank ='cuda'
        # create empty metric
        metrics = {}
        for _, key in enumerate(self.args.targets):
            metrics[key] = {}
            metrics[key]['vis0'] = IoUMetric(label_indices=self.cfg['label_indices'][key],
                                                min_visibility=None,
                                                max_visibility=None,
                                                target_class=key).to(rank)
            if key in {'vehicle', 'pedestrian'}:
                metrics[key]['vis40up'] = IoUMetric(label_indices=self.cfg['label_indices'][key],
                                                    min_visibility=2,
                                                    max_visibility=None,
                                                    target_class=key).to(rank)
                metrics[key]['vis40down'] = IoUMetric(label_indices=self.cfg['label_indices'][key],
                                                    min_visibility=None,
                                                    max_visibility=2,
                                                    target_class=key).to(rank)


        # set to evaluation mode
        self.mode_selection(isTrain=False)
        with torch.no_grad():
            for b, batch in enumerate(dataloader):
                batch = self.reform_batch(batch)
                pred_bev = self.model(batch, isTrain=False)

                for key, item in metrics.items():
                    for vis, metric in item.items():
                        metric.update(pred_bev[key][0], batch, label=None)

                self.print_validation_progress(b, d_len-1)
                
                min_idx, max_idx = 0, 50
                if self.args.visualization and b in range(min_idx, max_idx+1):
                    scenes = self.vis(batch, pred_bev)
                    scenes = np.array(scenes) # shape : (4, 314, 1636, 3)
                    image_printer(idx=b, bevs=scenes, save_path=self.args.save_dir, 
                                img_dir='bev_images', img_name='bev', rank=self.rank)

                #if b==10 : break
            if self.args.ddp: dist.barrier()
        
        IoU = {}
        for key, item in metrics.items():
            IoU[key] = {}
            for vis, metric in item.items():
                IoU[key][vis] = np.max(np.array([value for _, value in metric.compute().items()]))

        miou = {'vis0':[], 'vis40up':[], 'vis40down':[]}
        for key, item in IoU.items():
            for vis, score in item.items():
                miou[vis].append(score)
                
        if miou['vis40up'] == []:
            miou['vis40up'] = [0]
        if miou['vis40down'] == []:
            miou['vis40down'] = [0]
        for vis in miou:
            miou[vis] = np.mean(miou[vis])
            
        if self.rank == 0:
            self.log.info(f">> Eval | mIoU@vis0:{miou['vis0']:.4f}  mIoU@vis40up:{miou['vis40up']:.4f}  mIoU@vis40down:{miou['vis40down']:.4f}")
            for key, item in IoU.items():
                iou_str = []
                for vis, score in item.items():
                    iou_str.append(f"{key[:3]}@{vis}: {score:.4f}" )
                    self.monitor[f'val_iou_{key[:3]}_{vis}'] = score

                iou_str = "  ".join(iou_str)
                self.log.info(f"        | {iou_str}")
 
        
        self.monitor['val_miou_vis0'] = miou['vis0']
        self.monitor['val_miou_vis40up'] = miou['vis40up']

        if self.monitor['prev_miou'] < miou['vis0']:
            self.monitor['prev_miou'] = miou['vis0']

            if self.rank == 0:
                self.log.info(f">> ⭐ This is the BEST model !           ")
                
        if self.rank == 0:              
            if self.args.save_checkpoint:
                m = self.model.module if hasattr(self.model, "module") else self.model
                if hasattr(m, "backbone") and m.backbone is not None:
                    self.save_trained_network_params(m.backbone, name='backbone', e=e)
                self.save_trained_network_params(m.vt, name='vt', e=e)
    
    
    def get_dict_loss(self, loss_map):

        assert len(loss_map.shape)==4

        loss_dict = {}
        for i, key in enumerate(self.args.targets):
            loss_dict[key] = {'loss':loss_map[:,i].mean()}

        return loss_dict