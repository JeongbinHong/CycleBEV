#!/bin/bash

python -m torch.distributed.run --nproc_per_node=8 --master_port=20001 train.py \
--random_seed 2026 \
--memo "" \
--wandb 0 \
--wandb_project "" \
\
--ddp 1 \
--gpu_idx_ddp "0,1,2,3,4,5,6,7" \
--training_name "pre_icvt_decoder_eff2345_D1V4P20_H" \
--model_name "IVT_pretrain" \
--vt_model "CVT" \
\
--ivt_backbone "efficientnet-b4" \
--ivt_backbone_pretrain True \
\
--batch_size 4 \
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
--w_ivt_dri 0.05 \
--w_ivt_veh 0.2 \
--w_ivt_ped 1.0 \
\
--val_ratio 0.0 \
--val_step 4 \
--last_eval_nums 3 \
--remain_num_chkpts 2 \
