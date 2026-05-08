import os
import sys
import logging
import pickle
import traceback
import torch
import torch.distributed as dist
import wandb
from datetime import datetime
import argumentparser as ap
from utils.functions import *
from helper import load_datasetloader, load_solvers

def main(args):
    # logging setting
    logging.basicConfig(
        filename=args.save_dir + '/test.log',
        filemode="w",
        format='%(asctime)s %(levelname)s:%(message)s',
        level=logging.INFO,
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )
    logger = logging.getLogger(__name__)

    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setLevel(level=logging.DEBUG)
    logger.addHandler(consoleHandler)


    # DDP setting
    if (bool(args.ddp)):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx_ddp
        backend = 'nccl'
        dist_url = 'env://'
        rank = int(os.environ['RANK'])
        world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')) #int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method=dist_url, rank=rank, world_size=world_size)
        dist.barrier()
        if rank==0: print(f'DDP 사용')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
        rank, world_size, local_rank = 0, 1, 0
        print('DDP 아님')

    try:
        # 20250904
        cfg = config_update(read_config(), args)
        with open(os.path.join(args.save_dir, 'config_dict.pkl'), 'wb') as f:
            pickle.dump(cfg, f)

        if rank==0:
            # print training info
            print_training_info(cfg, args, logger)

        # dtype define
        _, float_dtype = get_dtypes()

        # prepare training data (0: train, 1: valid)
        dataset, dataloader, sampler = load_datasetloader(cfg=cfg,
                                                        args=args,
                                                        dtype=torch.FloatTensor,
                                                        world_size=world_size,
                                                        rank=local_rank,
                                                        mode='train'
                                                        )

        if rank==0:
            logger.info(f">> Number of available Train samples is [{len(dataset[0])}]")
            logger.info(f">> Number of available Val samples is [{len(dataset[1])}]")

        # define network
        solver = load_solvers(cfg, args, dataset[0].num_scenes, logger, float_dtype,
                              world_size=world_size, rank=local_rank, isTrain=False)

        if args.ddp==0 or (args.ddp and local_rank==0):
            if args.wandb:
                CFG = solver.wandb_tracker()
                run_epoch = run_wandb(args, wandb)
                run_epoch.config.update({"CFG":CFG['cfg'], "args":vars(args)})

        # ------------------------------------------
        # Evaluation
        # ------------------------------------------
        if (bool(args.ddp)):
            torch.cuda.synchronize()
            dist.barrier()

        solver.eval(dataset[1], dataloader[1], sampler[1], 1)

        if args.wandb and rank==0:
            CFG = solver.wandb_tracker()
            if args.model_name in {'Eff_UNet', 'ICVT_Decoder', 'ICVT_Decoder_skip', 'ICVT_Decoder_VT', 'ICVT_Decoder_FromPretrained'}:
                run_epoch.log({
                        "Train Loss": CFG['train_loss'],
                        "Valid mIoU@vis0": CFG['val_miou'],
                        "Valid IoU dri@visAll": CFG['val_iou_dri'],
                        "Valid IoU veh@visAll": CFG['val_iou_veh'],
                        "Valid IoU ped@visAll": CFG['val_iou_ped'],
                        }, step=1)
            else:
                run_epoch.log({
                        "Train Loss": CFG['train_loss'],
                        "Valid mIoU@vis0": CFG['val_miou_vis0'],
                        "Valid mIoU@vis40": CFG['val_miou_vis40'],

                        "Valid IoU dri@visAll": CFG['val_iou_dri_vis0'],
                        "Valid IoU veh@visAll": CFG['val_iou_veh_vis0'],
                        "Valid IoU veh@vis40%": CFG['val_iou_veh_vis40'],
                        "Valid IoU ped@visAll": CFG['val_iou_ped_vis0'],
                        "Valid IoU ped@vis40%": CFG['val_iou_ped_vis40'],

                        }, step=1)

        solver.init_loss_tracker()

        if rank==0: logger.info(f"The training has been completed 🚩")

    except Exception:
        logging.error(traceback.format_exc())

if __name__ == '__main__':

    args = ap.args
    seed_fixer(args.random_seed)

    now = datetime.now()
    current_time = now.strftime("%m-%d-%H-%M_")
    args.training_name = current_time + args.training_name

    if args.load_trained_vt:
        args.trained_vt_path = os.path.join(args.save_dir, args.trained_vt_path)

    args.save_dir = os.path.join(args.save_dir, args.training_name)
    if args.save_dir != '' and not os.path.exists(args.save_dir):
        try: os.makedirs(args.save_dir)
        except OSError: print(f'>> [{args.save_dir}] seems to already exist!!')
        args.load_pretrained = 0 # because there are no pre-trained nets in save_dir

    main(args)
