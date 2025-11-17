#!/bin/bash

python -m torch.distributed.run --nproc_per_node=4 --master_port=20001 train.py \
--memo "" \
--random_seed 2026 \
--wandb 0 \
--wandb_project "" \
\
--ddp 1 \
--gpu_idx_ddp "0,1,2,3" \
--training_name "baseline_PETRv2" \
--model_name "Baseline" \
--vt_model "PETR" \
\
--get_height 0 \
--multi_class_height_weights 0.0 0.0 0.0 0.0 \
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
--w_vt_height 0.0 \
\
--val_step 3 \
--save_checkpoint 1 \
--visualization 1 \