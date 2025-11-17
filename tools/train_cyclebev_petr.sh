#!/bin/bash

python -m torch.distributed.run --nproc_per_node=4 --master_port=20001 train.py \
--random_seed 2026 \
--memo "" \
--wandb 0 \
--wandb_project "" \
\
--ddp 1 \
--gpu_idx_ddp "0,1,2,3" \
--training_name "cyclebev_PETRv2" \
--model_name "CycleBEV" \
--vt_model "PETR" \
\
--pretrained_ivt_path "./saved_models/pretrained_cp/08-26-19-44_pre_icvt_decoder_eff2345_fpn_D1V4P20_H" \
\
--get_height 1 \
--multi_class_height_weights 0.1 0.25 1.0 5.0 \
\
--batch_size 2 \
--num_workers 6 \
\
--targets "drivable" "vehicle" "pedestrian" \
--visibility 0 \
--augmentation 0 \
\
--img_h 224 \
--img_w 480 \
--img_top_crop 46 \
\
--w_vt_dri 0.03125 \
--w_vt_veh 0.25 \
--w_vt_ped 1.0 \
--w_vt_height 1.0 \
\
--w_ivt_dri 0.05 \
--w_ivt_veh 0.2 \
--w_ivt_ped 1.0 \
\
--w_bev_loss 1.0 \
--w_pvcc_loss 0.4 \
--w_pvgg_loss 1.0 \
--w_feat_loss 0.001 \
--feat_loss_type "smooth_L1" \
\
--val_step 3 \
--save_checkpoint 1 \
--visualization 1 \
