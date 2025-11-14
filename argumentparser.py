import argparse

parser = argparse.ArgumentParser()

# ------------------------
# Exp Info
# ------------------------
parser.add_argument('--memo', type=str, default='')
parser.add_argument('--wandb', type=int, default=0, choices=[0,1], help='bool type')
parser.add_argument('--wandb_project', type=str, default="Your_wandb_project")
parser.add_argument('--training_name', type=str, default="Your_training_name")
parser.add_argument('--save_dir', type=str, default='./saved_models')

parser.add_argument('--gpu_idx', type=int, default=0)
parser.add_argument('--gpu_idx_ddp', type=str, default='0,1,2,3') # 0,1,2,3,4,5,6,7
parser.add_argument('--model_name', type=str, default='CycleBEV', choices=['Baseline','CycleBEV','IVT_pretrain'])
parser.add_argument('--ddp', type=int, default=0, choices=[0,1], help='bool type')
parser.add_argument('--drop_last', type=int, default=0, choices=[0,1], help='bool type')
parser.add_argument('--bool_mixed_precision', type=int, default=0, choices=[0,1], help='bool type')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--random_seed', type=int, default=2026)
parser.add_argument('--device', type=str, default='cuda')

# ------------------------
# Retrain model 
# ------------------------
parser.add_argument('--load_pretrained', type=int, default=0, choices=[0,1], help='bool type')
parser.add_argument('--load_trained_vt', type=int, default=0, choices=[0,1], help='bool type')
parser.add_argument('--trained_vt_path', type=str, default='')
parser.add_argument('--visualization', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=1)

# ------------------------
# Dataset
# ------------------------
parser.add_argument('--config_path', type=str, default='./config')
parser.add_argument('--dataset_type', type=str, default='nuscenes')
parser.add_argument('--val_ratio', type=float, default=0.0)
parser.add_argument('--augmentation', type=int, default=0, choices=[0,1])

parser.add_argument('--targets', type=str, nargs='*', default=['drivable', 'vehicle', 'pedestrian'], choices=['drivable', 'vehicle', 'pedestrian'])
parser.add_argument('--visibility', type=int, default=0, choices=[0,1], help='bool type. 1: vis>40%')

parser.add_argument('--img_h', type=int, default=224) 
parser.add_argument('--img_w', type=int, default=480) 
parser.add_argument('--img_top_crop', type=int, default=46)

parser.add_argument('--bev_w', type=int, default=200)
parser.add_argument('--bev_h', type=int, default=200)
parser.add_argument('--bev_resize_h', type=int, default=224)
parser.add_argument('--bev_resize_w', type=int, default=480)
parser.add_argument('--bev_w_meters', type=int, default=100)
parser.add_argument('--bev_h_meters', type=int, default=100)
parser.add_argument('--bev_offset', type=int, default=0)

parser.add_argument('--get_height', type=int, default=0, choices=[0,1])
parser.add_argument('--dri_height', type=float, default=1.0)
parser.add_argument('--multi_class_height_weights', type=float, nargs='*', default=[0.1, 0.25, 1.0, 5.0])

parser.add_argument('--past_horizon_seconds', type=float, default=0.5)
parser.add_argument('--future_horizon_seconds', type=float, default=0.0)
parser.add_argument('--target_sample_period', type=float, default=2.0)  # Hz ---
# parser.add_argument('--num_past_temporal_samples', type=int, default=3)
# parser.add_argument('--inference_temporal_sample_index', type=int, nargs='*', default=[-3,-2,-1])

# ------------------------
# IVT
# ------------------------
parser.add_argument('--pretrained_segmenter_path', type=str, default='') 
parser.add_argument('--pretrained_ivt_path', type=str, default='./saved_models/pretrained_cp/08-26-19-44_pre_icvt_decoder_eff2345_fpn_D1V4P20_H') 
parser.add_argument('--ivt_backbone', type=str, default='efficientnet-b4', choices=['resnet18', 'resnet50', 'resnet101', 'efficientnet-b4'])
parser.add_argument('--ivt_backbone_pretrain', type=bool, default=True)
parser.add_argument('--logit_processing', type=str, default='sigmoid', choices=['raw', 'sigmoid'])
parser.add_argument('--feat_loss_type', type=str, default='smooth_L1', choices=['L1', 'L2', 'mae', 'mse', 'cosim', 'smooth_L1'])

# ------------------------
# Training Env
# ------------------------
parser.add_argument('--vt_model', type=str, default='CVT', choices=['CVT', 'BEVFormer', 'LSS', 'PETR', 'IVT'])
parser.add_argument('--batch_size', type=int, default=2, help='batch size per each gpu')
parser.add_argument('--val_step', type=int, default=3)
parser.add_argument('--last_eval_nums', type=int, default=1)
parser.add_argument('--remain_num_chkpts', type=int, default=1)

parser.add_argument('--optimizer_type', type=str, default='adamw', help='support adam and adamw only')
parser.add_argument('--lr_schd_type', type=str, default='OnecycleLR', choices=['none', 'OnecycleLR'])

parser.add_argument('--bool_find_unused_params', type=int, default=0, choices=[0,1], help='bool type')
parser.add_argument('--iou_threshold', type=str, default='@0.50', help='@0.30, @0.35, @0.40, @0.45, @0.50, @0.55, @0.60')

parser.add_argument('--save_checkpoint', type=int, default=1)
parser.add_argument('--grad_clip', type=float, default=1.0)

# ------------------------
# Weights
# ------------------------
parser.add_argument('--w_vt_dri', type=float, default=0.03125)
parser.add_argument('--w_vt_veh', type=float, default=0.25)  
parser.add_argument('--w_vt_ped', type=float, default=1.0)
parser.add_argument('--w_vt_height', type=float, default=1.0)

parser.add_argument('--w_ivt_dri', type=float, default=0.05)
parser.add_argument('--w_ivt_veh', type=float, default=0.2)  
parser.add_argument('--w_ivt_ped', type=float, default=1.0)

parser.add_argument('--w_bev_loss', type=float, default=1.0)
parser.add_argument('--w_pvcc_loss', type=float, default=0.4)
parser.add_argument('--w_pvgg_loss', type=float, default=1.0)
parser.add_argument('--w_feat_loss', type=float, default=0.001)


args = parser.parse_args()
