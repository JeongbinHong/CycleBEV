import os.path
from utils.functions import *
from utils.loss import *
from utils.metrics import IoUMetric
from torch.nn.parallel import DistributedDataParallel as DDP
from models.extractors import EffNet_Extractor, ResNet_Extractor
from models.icvt.decoder import *
import torch.distributed as dist
from utils.visualizer import save_images
from einops import rearrange
from utils.functions import Normalize
from utils.loss import Optimizers
from utils.visualizer import BaseViz

class FullModel(nn.Module):
    def __init__(self, args, backbone, vt, norm):
        super().__init__()
        self.args = args
        if backbone is not None:
            self.backbone = backbone
        self.vt = vt
        self.norm = norm

    def forward(self, batch, isTrain=True):
        if self.args.vt_model in {'LSS', 'PETR'}:
            bev_pred, _ = self.vt(batch, isTrain)
        elif self.args.vt_model == 'CVT':
            img_features = self.backbone(self.norm(batch['image']))
            bev_pred, _ = self.vt(batch, img_features, isTrain)
        elif self.args.vt_model == 'BEVFormer':
            input_images = self.norm(batch['image'])
            if len(input_images.shape) > 4:
                img_features = [self.backbone(img) for img in input_images]
            else:
                img_features = self.backbone(input_images)
            bev_pred, _ = self.vt(batch, img_features, isTrain)
        else:
            raise ValueError(f"Unsupported vt_model: {self.args.vt_model}")

        return bev_pred
    
        
class Solver:

    def __init__(self, cfg, args, num_train_scenes, world_size=None, rank=None, logger=None, dtype=None, isTrain=True):
        seed_fixer(args.random_seed)
        self.save_dir = args.save_dir
        self.trained_vt_path = args.trained_vt_path

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
                        'val_miou_vis0': 0,
                        'val_miou_vis40up': 0,
                        'val_iou_dri_vis0': 0,
                        'val_iou_veh_vis0': 0,
                        'val_iou_veh_vis40up': 0,
                        'val_iou_ped_vis0': 0,
                        'val_iou_ped_vis40up': 0,
                        'prev_miou': 0
                        #'cur_lr': args.learning_rate
                        }
        
        self.save_target_output = []
        

        self.norm = Normalize('imagenet')
        # self.norm_ns = Normalize('nuscenes_topcrop')

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
            
        if self.args.load_trained_vt:
            if args.vt_model in {'LSS', 'PETR'}:
                backbone = None
            elif args.vt_model in {'CVT', 'BEVFormer'}:
                backbone = self.load_pretrained_model(backbone, self.trained_vt_path, 'backbone')
            vt = self.load_pretrained_model(vt, self.trained_vt_path, 'vt')
                                                                        
        model = FullModel(args, backbone, vt, self.norm)

        # DDP
        if args.ddp: 
            torch.cuda.set_device(rank)
            model = model.type(dtype).to(rank)
            self.model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        else:
            self.model = model.type(dtype).cuda()

        if len(self.args.targets)==3:
            assert self.args.targets == ['drivable', 'vehicle', 'pedestrian']

        self.class_id = {'drivable': 1,
                         'vehicle': 2,
                         'pedestrian': 3}
        
        self.weights = {'drivable': args.w_vt_dri,
                        'vehicle': args.w_vt_veh,
                        'pedestrian': args.w_vt_ped,
                        'height': args.w_vt_height
                        }
            
        min_visibility = None
        if args.visibility and ('vehicle' in self.args.targets or 'pedestrian' in self.args.targets):
            min_visibility = 2
        self.bev_loss = LossScratch(cfg=self.cfg, args=self.args, min_visibility=min_visibility)
        
        train_set = self.cfg[f'{args.vt_model}']['training']
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
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k.replace("module.", "") # removing ‘module.’ from key
                new_state_dict[name] = v

            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model.state_dict()}
            model.load_state_dict(pretrained_dict)

        if self.rank == 0:
            self.log.info('>> trained parameters are loaded from {%s}' % file_name)
            self.log.info(">> current training status : %.4f mIoU" % checkpoint['prev_miou'])

        return model


    def save_trained_network_params(self, model, name, e):
        # save trained model
        _ = save_read_latest_checkpoint_num(os.path.join(self.save_dir), e, isSave=True)
        file_name = self.save_dir + f'/saved_chk_point_{name}_{e}.pt'

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
            self.log.info(f"[Epoch {e:d}, {tl:.2f} hrs left] "
                          f"loss: {self.monitor['loss_sum']:.4f}, "
                          f"(cur lr: {cur_lr:.7f})")


    def print_training_progress(self, e, b, time): # real time loss
        if (self.rank == 0):
            if (b >= self.num_batches - 2): sys.stdout.write('\r')
            else: sys.stdout.write(f"\r [Epoch {e}] {b} / {self.num_batches} ({time:.4f} sec/sample), " 
                                   f"loss: {self.monitor['loss']:.4f}, "
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

    def train(self, batch, e, b):
        batch = self.reform_batch(batch)

        self.opt.zero_grad()
        pred = self.model(batch) 
        
        loss = torch.zeros(1).to(self.rank)
        loss_dict = self.bev_loss.main(pred, batch)
        for key, item in loss_dict.items():
            loss += self.weights[key] * item['loss']

        loss.backward()
        if self.args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.opt.step()

        self.monitor['loss'] = loss.item()
        self.monitor['loss_sum'] += loss.item()

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
        # if d_len > 0:
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
            self.log.info(f">> Eval | mIoU@vis0:{miou['vis0']:.4f}  mIoU@vis40up:{miou['vis40up']:.4f}  mIoU@vis40down:{miou['vis40down']:.4f} ✅")
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
