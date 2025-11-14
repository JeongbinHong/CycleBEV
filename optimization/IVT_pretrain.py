import os.path

from utils.functions import *
from utils.loss import *
from utils.metrics import IoUMetric
from models.icvt.icvt import InverseCrossViewTransformer
from models.icvt.decoder import *
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils.visualizer import save_images
from einops import rearrange
from utils.functions import Normalize
from utils.loss import Optimizers

class FullModel(nn.Module):
    def __init__(self, cfg, args, ivt, decoder):
        super().__init__()
        self.args = args
        self.ivt = ivt
        self.decoder = decoder
        
        self.noise_mode = cfg['IVT']['bev']['noise_mode']
        self.noise_std = cfg['IVT']['bev']['noise_std']

    def forward(self, batch):
        if self.noise_mode != 'none':
            noise_bev = batch['bev'].clone() + torch.randn_like(batch['bev']) * self.noise_std
            
            if self.noise_mode == 'hard':
                noise_bev = torch.clamp(noise_bev, 0, 1)

            elif self.noise_mode == 'soft':
                noise_bev = noise_bev.abs()
                excess = (noise_bev - 1).clamp(min=0)
                noise_bev = noise_bev - 2 * excess
                
            batch['bev'] = noise_bev
            
            
        if self.args.get_height:
            batch['bev'] = torch.concat([batch['bev'], batch['bev_height']], dim=1)
            
        ivt_features, _ = self.ivt(batch)
        pred_mask = self.decoder(ivt_features)

        return pred_mask

class Solver:

    def __init__(self, cfg, args, num_train_scenes, world_size=None, rank=None, logger=None, dtype=None, isTrain=True):
        self.save_dir = args.save_dir
        # load pre-trained settings or save current settings
        if (isTrain):
            if (args.load_pretrained == 1):
                if (os.path.exists(self.save_dir) is not True): sys.exit(f'>> path {self.save_dir} does not exist!!')
                else:
                    with open(os.path.join(self.save_dir, 'config.pkl'), 'rb') as f:
                        args = pickle.load(f)
                    args.load_pretrained = 1

            # or save current settings
            else:
                if (rank==0):
                    with open(os.path.join(self.save_dir, 'config.pkl'), 'wb') as f:
                        pickle.dump(args, f)

        # training setting
        self.args = args
        self.cfg = cfg
        self.rank, self.world_size = rank, world_size
        self.log = logger
        self.args.targets
        
        self.dtype = dtype
        if args.drop_last:
            self.num_batches = int(num_train_scenes / (args.batch_size * world_size))
        else:
            self.num_batches = math.ceil(num_train_scenes / (args.batch_size * world_size))

        # training monitoring
        self.monitor = {'iter': 0,
                        'loss_sum':0,
                        'val_miou': 0,
                        'val_iou_dri': 0,
                        'val_iou_veh': 0,
                        'val_iou_ped': 0,
                        'prev_miou': 0
                        #'cur_lr': args.learning_rate
                        }
        
        self.save_target_output = {'train':{},
                                   'val':{}}

        # self.norm = Normalize('imagenet')
        self.norm_ns = Normalize('nuscenes_topcrop')


        ivt_feat_shapes = self.cfg['IVT']['encoder']['ivt_feat_shapes']
        bev_layer_nums = self.cfg['IVT']['encoder']['bev_layer_nums']
        ivt_decoder_type = self.cfg['IVT']['decoder']['type']
        final_mask_h, final_mask_w = self.cfg['IVT']['image']['final_mask_h'], self.cfg['IVT']['image']['final_mask_w']
        
        if ivt_decoder_type=='IVTDecoder':
            decoder = IVTDecoder(in_channels=ivt_feat_shapes[0][1],
                                out_channels=len(args.targets)+1,  # with background
                                target_size=(final_mask_h, final_mask_w))
        elif ivt_decoder_type=='IVTSelfSkip2Decoder':
            ivt_feat_shapes = ivt_feat_shapes[:2]
            decoder = IVTSelfSkip2Decoder(in_channels=[shapes[1] for shapes in ivt_feat_shapes],
                                        out_channels=len(args.targets)+1,  # with background
                                        target_size=(final_mask_h, final_mask_w))

        # model define
        ivt = InverseCrossViewTransformer(self.cfg, self.args, ivt_feat_shapes, bev_layer_nums, rank=rank, from_pretrain=False)
        
        if isTrain == False:
            ivt = self.load_pretrained_model(ivt, args.pretrained_ivt_path, 'ivt')
            decoder = self.load_pretrained_model(decoder, args.pretrained_ivt_path, 'decoder')
            
        model = FullModel(self.cfg, args, ivt, decoder)

        # DDP
        if args.ddp: 
            torch.cuda.set_device(rank)
            model = model.type(dtype).to(rank)
            self.model = DDP(model, device_ids=[rank], find_unused_parameters=bool(args.bool_find_unused_params))
        else:
            self.model = model.type(dtype).cuda()

        self.iou_scores = {}
        self.class_id = {'drivable': 1,
                         'vehicle': 2,
                         'pedestrian': 3}
        self.target_nums = [self.class_id[target] for target in self.args.targets]
        for target in args.targets:
            self.iou_scores[target] = 0

        self.weights = {'drivable': args.w_ivt_dri,
                        'vehicle': args.w_ivt_veh,
                        'pedestrian': args.w_ivt_ped,
                        }
        
        train_set = self.cfg[f'IVT']['training']
        num_epochs = train_set['num_epochs']
        learning_rate = train_set['learning_rate']
        weight_decay = train_set['weight_decay']
        div_factor = train_set['div_factor']
        pct_start = train_set['pct_start']
        final_div_factor = train_set['final_div_factor']

        # define optimizer
        self.opt = Optimizers(self.model, optimizer_type=args.optimizer_type,
                             learning_rate=learning_rate, weight_decay=weight_decay).opt
        # training schedule
        if self.args.lr_schd_type != 'none':
            lr_cfg = {'max_lr':learning_rate, 'div_factor': div_factor,
                        'final_div_factor': final_div_factor, 'pct_start':pct_start,
                        'steps_per_epoch': self.num_batches, 'epochs':num_epochs
                        }
            self.lr_scheduler = LRScheduler(self.opt, type=self.args.lr_schd_type, config=lr_cfg)

        self.alpha = [self.cfg['Loss']['target_focal'][target]['alpha'] for target in self.args.targets]
        self.gamma = [self.cfg['Loss']['target_focal'][target]['gamma'] for target in self.args.targets]
        
        

        if (rank == 0):
            print(">> Optimizer is loaded from {%s} " % os.path.basename(__file__))


    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def mode_selection(self, isTrain=True):
        if (isTrain): 
            self.model.train()
        else: 
            self.model.eval()

    def init_loss_tracker(self):
        self.monitor['loss_sum'] = 0
        self.monitor['max_grad'] = 0


    def normalize_loss_tracker(self):
        self.monitor['loss_sum'] /= self.num_batches

    def lr_scheduler_step(self):
        if self.args.lr_schd_type != 'none':
            self.lr_scheduler()

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
        self.log.info(f">> ⭐ current network is saved ...")
        remove_past_checkpoint(os.path.join('./', self.save_dir), self.args.remain_num_chkpts, name)

    def print_status(self, e, tl):
        if (self.rank==0):
            cur_lr = self.opt.param_groups[0]['lr']
            self.log.info(f'[Epoch {e:d}, {tl:.2f} hrs left] '
                          f"loss: {self.monitor['loss_sum']:.5f}, "
                          f'(cur lr: {cur_lr:.7f})')

    def print_training_progress(self, e, b, time):
        if (self.rank == 0):
            if (b >= self.num_batches - 2): sys.stdout.write('\r')
            else: sys.stdout.write(f"\r [Epoch {e}] {b} / {self.num_batches} ({time:.4f} sec/sample), " 
                                   f"loss: {self.monitor['loss']:.5f}, "
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
            'val_iou_dri': self.monitor['val_iou_dri'],
            'val_iou_veh': self.monitor['val_iou_veh'],
            'val_iou_ped': self.monitor['val_iou_ped'],
        }
        return CFG

    # ------------------------
    # Training
    # ------------------------

    def reform_batch(self, batch):
        for key, value in batch.items():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].type(self.dtype).to(self.rank)
                if len(batch[key].shape)>=5 and batch[key].size(1)==1:
                    batch[key] = batch[key][:, 0] # remove temporal dimension

            if key in {'image','m2f_images'}: 
                batch[key] = rearrange(batch[key], 'b n c h w -> (b n) c h w')
            elif key in {'image_seg_masks'}:
                batch[key] = rearrange(batch[key], 'b n h w -> (b n) h w')
        return batch
    
    def train(self, batch, e, b):  # train.py - solver.train(data)  ->  batch = data
        batch = self.reform_batch(batch)

        target_mask = batch['image_seg_masks']
        target_mask = F.one_hot(target_mask.long(), num_classes=4)
        target_mask = target_mask.permute(0, 3, 1, 2).float()

        self.opt.zero_grad()
        pred_mask = self.model(batch) # [b, c, h, w]
        
        target_mask = target_mask[:,self.target_nums]
        pred_mask = pred_mask[:,1:]
        _loss =  [sigmoid_focal_loss(pred_mask[:,i], target_mask[:,i], self.alpha[i], self.gamma[i]) for i in range(pred_mask.size(1))]
        _loss = torch.stack(_loss, dim=1)
        _loss = self.get_dict_loss(_loss)

        loss = torch.zeros(1).to(self.rank)
        for key, item in _loss.items():
            loss += self.weights[key] * item['loss']

        # back-propagation
        loss.backward()
        if self.args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.opt.step()

        self.monitor['loss'] = loss.item()
        self.monitor['loss_sum'] += loss.item()

        # increase iteration number
        self.monitor['iter'] += 1

        #get_grad_norm(self.model, log=self.log, mode='mean')



    # ------------------------
    # Validation
    # ------------------------
    def eval(self, dataset, dataloader, sampler, e):
        if self.args.ddp:
            rank = dist.get_rank()
        else: 
            rank ='cuda'
        metrics = {}
        for _, key in enumerate(self.args.targets):
            metrics[key] = IoUMetric(label_indices=self.cfg['label_indices'][key],
                                    min_visibility=None,
                                    target_class=key).to(rank)


        # set to evaluation mode
        d_len = len(dataloader)
        self.mode_selection(isTrain=False)
        with torch.no_grad():
            for b, batch in enumerate(dataloader):
                batch = self.reform_batch(batch)

                m2f_images = self.norm_ns(batch['m2f_images'])
                target_mask = batch['image_seg_masks']
                target_mask = F.one_hot(target_mask.long(), num_classes=4)
                target_mask = target_mask.permute(0, 3, 1, 2).float()

                pred_mask = self.model(batch) # [B, C, H, W]

                target_mask = target_mask[:,[0]+self.target_nums]

                for i, (key, item) in enumerate(metrics.items()):
                    item.update(
                        pred_mask[:, i+1:i+2],
                        batch,
                        label=target_mask[:, i+1:i+2]
                    )
                min_idx, max_idx = 0, 50
                if b in range(min_idx, max_idx+1):
                    output_dir = os.path.join(self.args.save_dir, 'pv_images')
                    pred_mask = pred_mask.argmax(dim=1)
                    target_mask = target_mask.argmax(dim=1)
                    save_images(self.args, b, output_dir, m2f_images, pred_mask, target_mask, filepath=batch['filepath'], norm_type='nuscenes_topcrop', rank=self.rank)
                    
                self.print_validation_progress(b, d_len-1)

                #if b==10 : break
            if self.args.ddp: dist.barrier()
            
        IoU = {}
        for key, item in metrics.items(): # key : class label
            IoU[key] = item.compute()[self.args.iou_threshold]   
            
        miou = np.mean([item for _, item in IoU.items()])
        if self.rank == 0: 
            self.log.info(f">> Eval | mIoU{self.args.iou_threshold} : {miou:.4f} ✅               ")
            iou_str = "  ".join([f"{key}:{IoU[key]:.4f}" for key in IoU])
            self.log.info(f"        | {iou_str}")
                
        for key, score in IoU.items():
            self.monitor[f'val_iou_{key[:3]}'] = score

        self.monitor['val_miou'] = miou
        if self.monitor['prev_miou'] < miou:
            self.monitor['prev_miou'] = miou

            if self.rank == 0:
                if self.args.ddp:
                    self.save_trained_network_params(self.model.module.ivt, name='ivt', e=e)
                    self.save_trained_network_params(self.model.module.decoder, name='decoder', e=e)
                else:
                    self.save_trained_network_params(self.model.ivt, name='ivt', e=e)
                    self.save_trained_network_params(self.model.decoder, name='decoder', e=e)

                self.log.info(f"The BEST model has been saved👍")


    def get_dict_loss(self, loss_map):

        assert len(loss_map.shape)==4

        loss_dict = {}
        for i, key in enumerate(self.args.targets):
            loss_dict[key] = {'loss':loss_map[:,i].mean()}

        return loss_dict
