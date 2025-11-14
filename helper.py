from torch.utils.data import DataLoader
# from utils.functions import *
from torch.utils.data.distributed import DistributedSampler
from utils.collate import *
from torchvision import transforms

def load_datasetloader(cfg, args, dtype, world_size, rank, mode='train'):

    # config = read_json(path='./config/config.json')

    if (args.dataset_type == 'nuscenes'):
        if args.vt_model in {'BEVFormer', 'PETR'} and cfg['use_temporal']:
            from NuscenesDataset.loader_typeApro_temp import DatasetLoader
        else:
            from NuscenesDataset.loader_typeApro import DatasetLoader

    else:
        sys.exit("[Error] '%s' dataset is not supported !!" % args.dataset_type)

    train_transforms = transforms.Compose([
                            transforms.ToTensor()])
        
    test_transforms = transforms.Compose([
                        transforms.ToTensor()])


    seq_collate = None
    if mode == 'train':
        val_mode = 'val_in_train' if args.val_ratio > 0.0 else 'val'
        if (bool(args.ddp)):
            train_dataset = DatasetLoader(args=args, img_transforms=train_transforms, dtype=dtype, world_size=world_size, rank=rank, mode='train')
            val_dataset = DatasetLoader(args=args, img_transforms=test_transforms, dtype=dtype, world_size=world_size, rank=rank, mode=val_mode)    

            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, drop_last=args.drop_last)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, drop_last=args.drop_last)

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, collate_fn=seq_collate)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True, sampler=val_sampler, collate_fn=seq_collate)
        else:
            train_dataset = DatasetLoader(args=args, img_transforms=train_transforms, dtype=dtype, world_size=1, rank=0, mode='train')
            val_dataset = DatasetLoader(args=args, img_transforms=test_transforms, dtype=dtype, world_size=1, rank=0, mode=val_mode)

            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True, collate_fn=seq_collate)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, drop_last=True, collate_fn=seq_collate)
            train_sampler, val_sampler = None, None

        return (train_dataset, val_dataset), (train_dataloader, val_dataloader), (train_sampler, val_sampler)
    
    elif mode == 'val':
        test_dataset = DatasetLoader(args=args, img_transforms=test_transforms, dtype=dtype, world_size=1, rank=0, mode='val')   
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, drop_last=False, collate_fn=seq_collate)
        return test_dataset, test_dataloader, None

    else:
        sys.exit(f"[Error] '{mode}' mode is not supported !!")    


def load_solvers(cfg, args, num_train_scenes, logger, dtype, world_size=None, rank=None, isTrain=True):

    args.front_cam_only = False
    if args.model_name == 'Baseline':
        from optimization.Baseline import Solver
        
    elif args.model_name == 'CycleBEV':    
        from optimization.CycleBEV import Solver

    elif args.model_name == 'IVT_pretrain':   
        from optimization.IVT_pretrain import Solver

    else:
        sys.exit("[Error] There is no solver for '%s' !!" % args.model_name)
    return Solver(cfg, args, num_train_scenes, world_size, rank, logger, dtype, isTrain)
