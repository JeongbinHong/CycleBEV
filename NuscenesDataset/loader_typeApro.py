from utils.functions import *
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from torch.utils.data import Dataset
from utils.geometry import *
from NuscenesDataset.common import *
from torchvision import transforms
from nuscenes.utils.data_classes import Box
from shapely.geometry import MultiPolygon
from utils.augmentation import PhotoMetricDistortion, AffineTransform
import NuscenesDataset.nuscenes.nuscenes as nuscenes_module
import NuscenesDataset.nuscenes.utils.data_classes as dc
import os
import tempfile

from NuscenesDataset.agent import Agent
from NuscenesDataset.scene import Scene
from nuscenes.utils.geometry_utils import view_points
from torch.multiprocessing import Manager
from pycocotools import mask as mask_utils

class CustomNuScenes(NuScenes):
    def __init__(self, version, dataroot, verbose=False, custom_table_names=[]):
        super().__init__(version=version, dataroot=dataroot, verbose=verbose)
        
        for custom_table_name in custom_table_names:
            custom_json_path = os.path.join(dataroot, version, f"{custom_table_name}.json")

            if os.path.exists(custom_json_path):
                with open(custom_json_path, "r", encoding="utf-8") as f:
                    custom_data = json.load(f)

                self.__dict__[custom_table_name] = custom_data 
                self.table_names.append(custom_table_name)
                self._token2ind[custom_table_name] = {entry["sample_token"]: idx for idx, entry in enumerate(custom_data)}
            else:
                raise FileNotFoundError(f"{custom_json_path} 파일을 찾을 수 없습니다.")

class DatasetLoader(Dataset):

    def __init__(self, args, img_transforms, dtype, world_size=None, rank=None, mode='train'):

        #random.seed(1024)

        self.mode = mode
        split = 'train' if mode in {'train', 'val_in_train'} else 'val' # 기존 train은 train-val로 split하고, 원본 val을 test로 사용
        self.args, self.dtype = args, dtype
        self.cfg = config_update(read_config(path=args.config_path), args)
        self.cfg['nuscenes']['dataset_dir']
        self.obs_len, self.pred_len = self.cfg['obs_len'], self.cfg['pred_len']
        self.seq_len = self.obs_len + self.pred_len

        ori_dims = (self.cfg['original_image']['w'], self.cfg['original_image']['h'])
        resize_dims = (self.cfg['image']['w'], self.cfg['image']['h']+self.cfg['image']['top_crop'])
        crop = (0, self.cfg['image']['top_crop'], resize_dims[0], resize_dims[1]) # top 46px
        self.img_aug_params = {'scale_width': resize_dims[0]/ori_dims[0],
                               'scale_height': resize_dims[1]/ori_dims[1],
                               'resize_dims': resize_dims,
                               'crop': crop}

        mask_h, mask_w = self.cfg['IVT']['image']['mask_h'], self.cfg['IVT']['image']['mask_w'] 
        final_mask_h, final_mask_w = self.cfg['IVT']['image']['final_mask_h'], self.cfg['IVT']['image']['final_mask_w'] 
        mask_top_crop = self.cfg['IVT']['image']['mask_top_crop']
        
        resize_dims2 = (mask_w, mask_h + mask_top_crop) # (960, 448+92)
        crop2 = (0, mask_top_crop, resize_dims2[0], resize_dims2[1]) # (0, 92, 960, 540) (left, top, right, bottom)
        self.img2_aug_params = {'scale_width': resize_dims2[0]/ori_dims[0],
                                'scale_height': resize_dims2[1]/ori_dims[1],
                                'resize_dims': resize_dims2,
                                'crop': crop2} # (900, 1600) -> (540, 960) -> (448, 960)
 
        self.img2_aug_params2 = {'scale_width': final_mask_w/mask_w,
                                'scale_height': final_mask_h/mask_h,
                                'resize_dims': (final_mask_w, final_mask_h),
                                'crop': None} # (448, 960) -> (224, 480)

        # Normalising input images (note: CVT normalizes images in the model)
        self.img_transforms = img_transforms
        self.img2_transforms = transforms.Compose([transforms.ToTensor()])
        
        # TODO : image augmentation should be here (for ablation study)
        if self.args.augmentation:
            from utils.augmentation import ImageAugmentation
            final_dim = (self.cfg['image']['h'], self.cfg['image']['w'])
            resize_lim = [0.8, 1.2]
            data_aug_conf = {'crop_offset': int(final_dim[0] * (1 - resize_lim[0])),
                            'resize_lim': resize_lim,
                            'final_dim': final_dim}
            self.img_aug = ImageAugmentation(data_aug_conf=data_aug_conf)

        # Bird's-eye view parameters
        self.bev_resolution, self.bev_start_position, self.bev_dimension = \
            calculate_birds_eye_view_parameters(self.cfg['lift']['x_bound'],
                                                self.cfg['lift']['y_bound'],
                                                self.cfg['lift']['z_bound'],
                                                isnumpy=True)


        # Nuscenes
        self.nusc_map = {}
        for k, v in enumerate(MAP_NAMES):
            self.nusc_map.update({v: NuScenesMap(dataroot=self.cfg['nuscenes']['dataset_dir'], map_name=v)})

        version=self.cfg['nuscenes']['version']
        dataroot=self.cfg['nuscenes']['dataset_dir']
        self.nusc = CustomNuScenes(version=version, 
                                    dataroot=dataroot, verbose=False,
                                    custom_table_names=["image_segment_annotation_rle"])
        
        with open(os.path.join(dataroot, version, "image_segment_annotation_meta.json"), "r", encoding="utf-8") as f:
            self.img_seg_mask_meta = json.load(f)
            

        # nuscenes map api
        from NuscenesDataset.map import Map
        self.hdmap = Map(dataroot, self.nusc)  

        # Splits into train/valid/test
        self.target_scenes = self.get_split(split)
        self.sample_records = self.return_ordered_sample_records()  

        seq_sample_indices = self.return_seq_sample_indices()
        random.shuffle(seq_sample_indices)

        if mode in {'train', 'val_in_train'}:
            num_val_scenes = int(len(seq_sample_indices) * self.args.val_ratio) # val
            num_train_scenes = len(seq_sample_indices) - num_val_scenes # train = all - val
            train_scenes = seq_sample_indices[:num_train_scenes]
            val_scenes = []
            for r in range(world_size):
                val_scenes += seq_sample_indices[num_train_scenes:]
            random.shuffle(val_scenes)
            self.scenes = train_scenes if (mode == 'train') else val_scenes
        else:
            self.scenes = seq_sample_indices
        self.num_scenes = len(self.scenes)

        if (rank==0):
            print(">> Dataset is loaded from {%s} " % os.path.basename(__file__))
            print(f'>> Number of available {mode} samples is {self.num_scenes}')

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx):
        seq_indices = self.scenes[idx]
        seq_indices = np.array(seq_indices)[self.cfg['target_frame_indices']]
        data = self.extract_seqdata_from_sample_records(seq_indices)
        return data



    def next_sample(self, seq_index):

        seq_indices = self.scenes[seq_index]
        seq_indices = np.array(seq_indices)[self.cfg['target_frame_indices']]
        data = self.extract_seqdata_from_sample_records(seq_indices)

        return data

    def extract_seqdata_from_sample_records(self, seq_indices):

        '''
        images (1 x n x c x h x w, tensor, float32) : normalized to 0~1 and (maybe) by mean and var
        intrinsics (1 x n x 3 x 3, tensor, float32) : crop and scaling are reflect
        extrinsics (1 x n x 4 x 4, tensor, float32) : egolidar_to_camera
        bev (1 x 12 x h x w, tensor, float32)
        center (1 x 1 x h x w, tensor, float32)
        visibility (1 x 1 x h x w, tensor, uint8)
        w2e, e2w (1 x 4 x 4, tensor, float32)       : World_to_Ego  Ego_to_World 

        ** BEV images are flipped upside-down and left-right.
        Need to apply np.flipud(np.fliplr(bev)) in order to match ego-centric frame (forward-up, side-left) **
        '''


        # Data corresponding to current time is extracted here fore semantic segmentation
        instance_map = {}
        seq_images, seq_intrinsics, seq_extrinsics, seq_bev, seq_bev_height, seq_bev_multi, seq_center, seq_visibility, \
            seq_instance, seq_offsets, seq_c2e, seq_e2w, seq_w2e = [], [], [], [], [], [], [], [], [], [], [], [], []
        intrinsic_4x4, extrinsics_4x4, lidar2img, sample_idx, timestamps, filenames = [], [], [], [], [], []  
        seq_m2f_images, seq_intrinsics2, seq_img_visibility = [], [], []
        seq_filepath, seq_timestamp = [], []
        seq_seg_masks = []

        for _, i in enumerate(seq_indices):

            if (i != seq_indices[self.obs_len-1]):
                continue

            rec = self.sample_records[i]

            # Camera data
            images, intrinsics, extrinsics, c2e, e2w, w2e, filepath, timestamp, m2f_images, intrinsics2, cam_tokens, img_seg_masks = self.return_input_data(rec) 

            seq_images.append(images)         # images.shape : [1,6,3,224,480]
            seq_intrinsics.append(intrinsics) # intrinsics.shape : [1,6,3,3]
            seq_extrinsics.append(extrinsics) # extrinsics.shape : [1,6,4,4]
            seq_c2e.append(c2e)  # C2E
            seq_e2w.append(e2w[None])  # EL2W
            seq_w2e.append(w2e[None])  # W2EL

            seq_m2f_images.append(m2f_images)
            seq_intrinsics2.append(intrinsics2)

            # BEV data
            data, instance_map = self.return_bev_labels(rec, instance_map) 
            bev = torch.flip(data['bev'], dims=(2, 3)) 
            
            center = torch.cat((data['aux']['center_score_veh'], data['aux']['center_score_ped']), dim=1) # 1 2 h w  
            center = torch.flip(center, dims=(2, 3))   
            visibility = data['visibility']
            visibility = torch.flip(visibility, dims=(2, 3))  

            seq_bev.append(bev)
            seq_center.append(center)
            seq_visibility.append(visibility)
            
            seq_seg_masks.append(img_seg_masks)
            
            if data['bev_height'] is not None:
                seq_bev_height.append(torch.flip(data['bev_height'], dims=(2, 3)))
            
            if data['bev_multi'] is not None:
                seq_bev_multi.append(torch.flip(data['bev_multi'], dims=(2, 3)))
                
        seq_bev = torch.cat(seq_bev, dim=0)  
        
        if len(seq_bev_height) != 0:
            seq_bev_height = torch.cat(seq_bev_height, dim=0)  
        else:
            seq_bev_height = torch.tensor(seq_bev_height)
            
        if len(seq_bev_multi) != 0:
            seq_bev_multi = torch.cat(seq_bev_multi, dim=0)  
        else:
            seq_bev_multi = torch.tensor(seq_bev_multi)

        seq_center = torch.cat(seq_center, dim=0)  
        seq_visibility = torch.cat(seq_visibility, dim=0)  
        seq_images = torch.cat(seq_images, dim=0)  
        seq_intrinsics = torch.cat(seq_intrinsics, dim=0) 
        seq_extrinsics = torch.cat(seq_extrinsics, dim=0) 
        seq_c2e = torch.cat(seq_c2e, dim=0) 
        seq_e2w = torch.cat(seq_e2w, dim=0) 
        seq_w2e = torch.cat(seq_w2e, dim=0) 

        seq_m2f_images = torch.cat(seq_m2f_images, dim=0) 
        seq_intrinsics2 = torch.cat(seq_intrinsics2, dim=0) 
        
        seq_seg_masks =  torch.cat(seq_seg_masks, dim=0) 

        return {'image': seq_images,
                'intrinsics': seq_intrinsics,
                'extrinsics': seq_extrinsics,
                'bev': seq_bev,
                'bev_height': seq_bev_height,
                'bev_multi': seq_bev_multi,
                'center': seq_center,
                'visibility': seq_visibility,
                'c2e' : seq_c2e,
                'e2w' : seq_e2w,
                'w2e' : seq_w2e,

                'm2f_images': seq_m2f_images,
                'intrinsics2': seq_intrinsics2,
                'image_seg_masks': seq_seg_masks,
                'sample_tokens': cam_tokens,
                'filepath': filepath,
                'timestamp': timestamp

                }
        


    def return_ordered_sample_records(self):
        '''
        Based on 'https://github.com/wayveai/fiery'
        '''

        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.target_scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples


    def return_seq_sample_indices(self):
        '''
        Based on 'https://github.com/wayveai/fiery'
        '''

        indices = []
        for index in range(len(self.sample_records)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.seq_len):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.sample_records):
                    is_valid_data = False
                    break
                rec = self.sample_records[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return indices


    def return_input_data(self, sample_record):
        '''
        egolidar : ego-pose based on lidar sensor
        egocam : ego-pose based on camera sensor
        '''
        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP']) # def get(self, table_name: str, token: str) -> dict:
        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])
        top_crop, left_crop = self.img_aug_params['crop'][1], self.img_aug_params['crop'][0]
        top_crop2, left_crop2 = self.img2_aug_params['crop'][1], self.img2_aug_params['crop'][0]

        # note : Fiery sets 'flat=True'
        EL2W = get_pose(egolidar['rotation'], egolidar['translation'], flat=True) # EgoLidar_to_World  # get_pose: from NuscenesDataset.common import *
        W2EL = get_pose(egolidar['rotation'], egolidar['translation'], flat=True, inv=True)  # World_to_EgoLidar

        images, intrinsics, extrinsics = [], [], []
        filepath, timestamp = [], []
        m2f_images, intrinsics2, img_visibilities = [], [], []
        cam_tokens = []
        img_seg_masks = []
        c2e = []
        for cam in CAMERAS: 
            '''
            from NuscenesDataset.common import *
            CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
            '''
            cam_token = sample_record['data'][cam]
            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
            filepath.append(self.cfg['nuscenes']['dataset_dir'] + cam_record['filename'])
            timestamp.append(cam_record['timestamp'])
            
            img_seg_mask = self.nusc.get('image_segment_annotation_rle', cam_token)
            img_seg_mask = self.rle_decode_mask(img_seg_mask)
            
            # Intrinsic parameter
            intrinsic = torch.from_numpy(np.array(cam['camera_intrinsic'])) # 3 x 3
            intrinsic = update_intrinsics(intrinsic, top_crop, left_crop,   
                                          scale_width=self.img_aug_params['scale_width'],
                                          scale_height=self.img_aug_params['scale_height'])
                                        
            # update 250204
            # intrinsic2 : Parameters tailored to the size of the image that IVT reconstructs
            intrinsic2 = update_intrinsics(intrinsic, top_crop2, left_crop2, 
                                          scale_width=self.img2_aug_params['scale_width'],
                                          scale_height=self.img2_aug_params['scale_height'])
            intrinsic2 = update_intrinsics(intrinsic2, 0, 0,   
                                          scale_width=self.img2_aug_params2['scale_width'],
                                          scale_height=self.img2_aug_params2['scale_height'])

            # Extrinsic parameter
            EC2C = get_pose(cam['rotation'], cam['translation'], flat=False, inv=True) # Egocam_to_Cam
            W2EC = get_pose(egocam['rotation'], egocam['translation'], flat=False, inv=True) # World_to_Egocam
            EL2C = EC2C @ W2EC @ EL2W # EgoLidar to Cam , note : Fiery uses C2EL
            extrinsic = torch.from_numpy(EL2C) # 4 x 4
            
            C2E = self.make_C2E(cam['rotation'], cam['translation']) # Cam to EgoVehicle
            C2E = torch.from_numpy(C2E)
            # E2C = C2E.inverse()  # EgoVehicle to Cam

            # Surround Image
            img = Image.open(Path(self.nusc.get_sample_data_path(cam_token)))
            img = resize_and_crop_image(img, resize_dims=self.img_aug_params['resize_dims'],
                                        crop=self.img_aug_params['crop'])

            # update 250204
            img2 = resize_and_crop_image(img, resize_dims=self.img2_aug_params['resize_dims'],
                                        crop=self.img2_aug_params['crop'])
            
            # TODO : image augmentation should be here
            if (self.mode == 'train' and self.args.augmentation and np.random.rand(1) < 0.5):
                img, intrinsic = self.img_aug(img, intrinsic)
            # TODO ---------
            
            # update 250204
            anns_dynamic = self.get_ann_rec_by_category(sample_record, DYNAMIC)
            anns_dynamic_all = []
            for anns_list in anns_dynamic: anns_dynamic_all += anns_list

            img = self.img_transforms(img) # c x h x w
            img2 = self.img2_transforms(img2)
            
            m2f_images.append(img2.unsqueeze(0).unsqueeze(0))

            images.append(img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            intrinsics2.append(intrinsic2.unsqueeze(0).unsqueeze(0))
            extrinsics.append(extrinsic.unsqueeze(0).unsqueeze(0))
            cam_tokens.append(cam_token)
            c2e.append(C2E.unsqueeze(0).unsqueeze(0))
            
            img_seg_masks.append(img_seg_mask.unsqueeze(0).unsqueeze(0))

        images = torch.cat(images, dim=1)
        intrinsics = torch.cat(intrinsics, dim=1)
        intrinsics2 = torch.cat(intrinsics2, dim=1)
        extrinsics = torch.cat(extrinsics, dim=1)
        c2e = torch.cat(c2e, dim=1)
        
        m2f_images = torch.cat(m2f_images, dim=1)             # [1, 6, 3, h, w]
        img_seg_masks = torch.cat(img_seg_masks, dim=1)
        
        return images, intrinsics, extrinsics, c2e, torch.from_numpy(EL2W), torch.from_numpy(W2EL), filepath, timestamp, m2f_images, intrinsics2, cam_tokens, img_seg_masks
    
    def make_C2E(self, rot, tran):
        w, x, y, z = rot
        # R = Quaternion(w, x, y, z).rotation_matrix  # (3,3)
        R = Quaternion(rot).rotation_matrix  # (3,3)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = np.array(tran)
        return T

    def rle_decode_mask(self, rle):
        # rle["size"] = self.img_seg_mask_meta['meta_data']['image_size']
        mask = np.zeros(rle["size"], dtype=np.uint8)
        
        # rle decoding
        class_ids = {
            'drivable': 1,
            'vehicle': 2,
            'pedestrian': 3
        }
        
        for class_name, rle_counts in rle["class_rle"].items():
            rle_data = {
                "size": rle["size"],
                "counts": rle_counts.encode("utf-8")
            }

            binary_mask = mask_utils.decode(rle_data)
            mask[binary_mask == 1] = class_ids[class_name]
        
        return torch.from_numpy(mask)

    def return_bev_labels(self, sample_record, instance_map):
        '''
        Based on https://github.com/bradyz/cross_view_transformers
        '''

        scene_token = sample_record['scene_token']
        scene_record = self.nusc.get('scene', scene_token)
        location = self.nusc.get('log', scene_record['log_token'])['location']
        lidar_sample = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        egopose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token']) # token, timestmap, rotation, translation

        # Raw annotations
        anns_dynamic = self.get_ann_rec_by_category(sample_record, DYNAMIC)

        # BEV images
        static = self.get_static_layers(location, egopose, STATIC)    # 200 x 200 x 2
        dividers = self.get_line_layers(location, egopose, DIVIDER)   # 200 x 200 x 2
        dynamic, height = self.get_dynamic_layers(anns_dynamic, egopose, get_height=self.args.get_height, dri_height=self.args.dri_height)
        bev = np.concatenate((static, dividers, dynamic), -1) # 200 x 200 x 12
        
        if self.args.get_height:
            max_height = np.max(height, axis=-1)
            bev_height = np.expand_dims(max_height, axis=-1)

            road = np.concatenate((static, dividers), axis=-1).any(axis=-1, keepdims=True)
            road_mask = (bev_height < 0) & (road == 1)
            bev_height[road_mask] = 0.0 # background=-1.0, drivable==0.0, vehicle&pedestrian=(0.0,5.0]
                                        # We assume that the maximum height of the object does not exceed 5m.
            
            # normalize by 6m
            bev_height += self.args.dri_height # background=0.0, drivable==1.0,   vehicle&pedestrian=(1.0,  6.0]
            bev_height /= self.args.dri_height+5.0 # background=0.0, drivable==0.167, vehicle&pedestrian=(0.167,1.0] # normalize by 5+1m
            bev_height = np.clip(bev_height, 0.0, 1.0)
            
        else:
            bev_height = None
            
        # if self.args.multi_class_layer: # background(0), drivable-area(1), vehicle(2), pedestrian(3)
        if self.args.targets == ["drivable", "vehicle", "pedestrian"]:
            target_bev = self.return_target_bev_label(bev, self.cfg['label_indices'])
            target_bev[:, :, 1] *= 2 # vehicle
            target_bev[:, :, 2] *= 3 # pedestrian
            bev_tartget_multi = np.max(target_bev, axis=-1) # [H, W]
            bev_tartget_multi = np.expand_dims(bev_tartget_multi, axis=-1) # [H, W, 1]
        else:
            bev_tartget_multi = None


        # Data for auxillary tasks, update 231006
        anns_dynamic_all = []
        for anns_list in anns_dynamic: anns_dynamic_all += anns_list
        _aux, visibility_veh, visibility_ped, instance_map = self.get_dynamic_objects(anns_dynamic_all, egopose, instance_map)

        # update 231006
        bev = torch.from_numpy(bev).permute(2, 0, 1).unsqueeze(0).contiguous() # 1 x 1 x 12 x h x w, float   # [[[[0,0,0,1,1,0,1, ...]]]]
        if bev_height is not None:
            bev_height = torch.from_numpy(bev_height).permute(2, 0, 1).unsqueeze(0).contiguous()
        if bev_tartget_multi is not None:
            bev_tartget_multi = torch.from_numpy(bev_tartget_multi).permute(2, 0, 1).unsqueeze(0).contiguous()
        aux = {}
        for key, value in _aux.items():
            aux[key] = torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).contiguous() # 1 x 1 x c x h x w
            '''
            - aux -
            segmentation 
            center_ohw 
            center_score_veh 
            center_score_ped 
            center_offset_veh 
            center_offset_ped 
            instance_veh 
            '''

        visibility_veh = torch.from_numpy(visibility_veh).unsqueeze(0).unsqueeze(0) # 1 x 1 x h x w, float
        visibility_ped = torch.from_numpy(visibility_ped).unsqueeze(0).unsqueeze(0)  # 1 x 1 x h x w, float
        visibility = torch.cat((visibility_veh, visibility_ped), dim=1)

        data = {'bev': bev,
                'bev_height': bev_height,
                'bev_multi': bev_tartget_multi,
                'aux': aux,
                'visibility': visibility,
                'location': location,
                'egopose': egopose}

        return data, instance_map


    def get_split(self, split):
        split_dir = Path(__file__).parent / 'nuscenes/splits'
        split_path = split_dir / f'{split}.txt'
        return split_path.read_text().strip().split('\n')


    def get_ann_rec_by_category(self, sample, categories):
        result = [[] for _ in categories]

        for ann_token in self.nusc.get('sample', sample['token'])['anns']:
            a = self.nusc.get('sample_annotation', ann_token)
            idx = self.get_category_index(a['category_name'], categories)

            if idx is not None:
                result[idx].append(a)

        return result


    def get_category_index(self, name, categories):
        """
        human.pedestrian.adult
        """
        tokens = name.split('.')

        for i, category in enumerate(categories):
            if category in tokens:
                return i

        return None


    def get_dynamic_layers(self, anns_by_category, egopose, get_height=False, dri_height=1.0):

        # egopose (lidar)
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse

        # bev center/resolution
        bev_center = - self.bev_start_position[:2] + 0.5 * self.bev_resolution[:2]
        bev_res = self.bev_resolution[:2]

        result, result_h = [], []
        for anns in anns_by_category:
            render = np.zeros((self.cfg['bev']['h'], self.cfg['bev']['w']), dtype=np.uint8)
            render_h = np.full((self.cfg['bev']['h'], self.cfg['bev']['w']), -dri_height, dtype=np.float32)
            for ann in anns:
                box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                box.translate(trans)
                box.rotate(rot)

                pts = box.bottom_corners()[:2].T  
                pts = np.round((pts + bev_center) / bev_res).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                
                cv2.fillPoly(render, [pts], 1.0, INTERPOLATION)
                if get_height:
                    cv2.fillPoly(render_h, [pts], ann['size'][-1], INTERPOLATION)
                    
            result.append(render)
            result_h.append(render_h)
            
        dynamic = np.stack(result, -1).astype('float32')
        obj_height = np.stack(result_h, -1).astype('float32')

        return dynamic, obj_height
    



    def get_static_layers(self, location, egopose, layers, patch_radius=150):  

        # egopose
        trans = -np.array(egopose['translation'])[:2]
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0] # yaw : 지면과 수직인 회전축 (좌우회전)
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse.rotation_matrix[:2, :2] # rotation을 새롭게 정의?

        # bev center/resolution
        bev_center = - self.bev_start_position[:2] + 0.5 * self.bev_resolution[:2]
        bev_res = self.bev_resolution[:2]

        pose = get_pose(egopose['rotation'], egopose['translation'], flat=True)
        x, y = pose[0][-1], pose[1][-1]
        box_coords = (x - patch_radius, y - patch_radius, x + patch_radius, y + patch_radius)
        records_in_patch = self.nusc_map[location].get_records_in_patch(box_coords, layers, 'intersect')

        result = list()
        for layer in layers:  # layers = STATIC = ['lane', 'road_segment', 'ped_crossing', 'walkway', 'stop_line', 'carpark']
            render = np.zeros((self.cfg['bev']['h'], self.cfg['bev']['w']), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map[location].get(layer, r)

                if layer == 'drivable_area': polygon_tokens = polygon_token['polygon_tokens']
                else: polygon_tokens = [polygon_token['polygon_token']]

                for p in polygon_tokens:
                    polygon = self.nusc_map[location].extract_polygon(p)
                    polygon = MultiPolygon([polygon])

                    exteriors = [np.array(poly.exterior.coords).T for poly in polygon.geoms] # 2 x N   # N=6
                    exteriors = [rot @ (p.T + trans).T for p in exteriors] # 2 x N
                    exteriors = [np.round((p.T + bev_center) / bev_res).astype(np.int32) for p in exteriors] # N x 2
                    exteriors = [np.fliplr(p) for p in exteriors]  # N x 2

                    cv2.fillPoly(render, exteriors, 1, INTERPOLATION) 


                    interiors = [np.array(pi.coords).T for poly in polygon.geoms for pi in poly.interiors]
                    interiors = [rot @ (p.T + trans).T for p in interiors]
                    interiors = [np.round((p.T + bev_center) / bev_res).astype(np.int32) for p in interiors]
                    interiors = [np.fliplr(p) for p in interiors]  # N x 2


                    cv2.fillPoly(render, interiors, 0, INTERPOLATION)

            result.append(render)
        return np.stack(result, -1).astype('float32')


    def get_line_layers(self, location, egopose, layers, patch_radius=150, thickness=2):

        # egopose
        trans = -np.array(egopose['translation'])[:2]
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse.rotation_matrix[:2, :2]

        # bev center/resolution
        bev_center = - self.bev_start_position[:2] + 0.5 * self.bev_resolution[:2]
        bev_res = self.bev_resolution[:2]

        pose = get_pose(egopose['rotation'], egopose['translation'], flat=True)
        x, y = pose[0][-1], pose[1][-1]
        box_coords = (x - patch_radius, y - patch_radius, x + patch_radius, y + patch_radius)
        records_in_patch = self.nusc_map[location].get_records_in_patch(box_coords, layers, 'intersect')

        result = list()
                                                     
        for layer in layers:  # layers = DIVIDER = ['road_divider', 'lane_divider']  # /NuscenesDataset/common.py
            render = np.zeros((self.cfg['bev']['h'], self.cfg['bev']['w']), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map[location].get(layer, r)
                line = self.nusc_map[location].extract_line(polygon_token['line_token'])

                p = np.float32(line.xy)    # 2 x N
                p = rot @ (p.T + trans).T  # 2 x N
                p = np.round((p.T + bev_center) / bev_res).astype(np.int32) # N x 2
                p = np.fliplr(p) # N x 2

                cv2.polylines(render, [p], False, 1, thickness=thickness)

            result.append(render)

        return np.stack(result, -1).astype(np.float32)


    def get_dynamic_objects(self, anns, egopose, ins_map):

        # egopose
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse

        # bev center/resolution
        h, w = self.cfg['bev']['h'], self.cfg['bev']['w']
        bev_center = - self.bev_start_position[:2] + 0.5 * self.bev_resolution[:2] # [50, 50]
        bev_res = self.bev_resolution[:2] # [0.5, 0.5]

        segmentation = np.zeros((h, w), dtype=np.uint8)
        center_ohw = np.zeros((h, w, 4), dtype=np.float32)

        # update 231006
        center_score_veh = np.zeros((h, w), dtype=np.float32)
        center_score_ped = np.zeros((h, w), dtype=np.float32)
        center_offset_veh = np.zeros((h, w, 2), dtype=np.float32)
        center_offset_ped = np.zeros((h, w, 2), dtype=np.float32)
        visibility_veh = np.full((h, w), 255, dtype=np.uint8)
        visibility_ped = np.full((h, w), 255, dtype=np.uint8)
        instance_veh = np.zeros((h, w), dtype=np.uint8)
        instance_ped = np.zeros((h, w), dtype=np.uint8)

        sigma = 1
        buf = np.zeros((h, w), dtype=np.uint8)
        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)
        i = 0
        for ann in anns: # annotations len=32
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            box.translate(trans)
            box.rotate(rot) 

            p = box.bottom_corners()[:2].T
            p = np.round((p + bev_center) / bev_res).astype(np.int32)
            p[:, [1, 0]] = p[:, [0, 1]] # 4 x 2

            center = np.round((box.center[:2] + bev_center) / bev_res).astype(np.int32).reshape(1, 2) # e.g [67, 112]
            center = np.fliplr(center)  # e.g [112, 67]

            buf.fill(0)
            cv2.fillPoly(buf, [p], 1, INTERPOLATION)
            mask = buf > 0

            if not np.count_nonzero(mask):
                continue

            # segmentation up
            segmentation[mask] = 255

            # instance map up
            if ann['instance_token'] not in ins_map:
                ins_map[ann['instance_token']] = len(ins_map) + 1
            ins_id = ins_map[ann['instance_token']]

            # update 231006
            if ('vehicle' in ann['category_name']):
                visibility_veh[mask] = ann['visibility_token']   # 0-40%, 40-60%, 60-80% and 80-100%. (1,2,3,4)
                instance_veh[mask] = ins_id
                center_offset_veh[mask] = center - coords[mask]
                center_score_veh[mask] = np.exp(-(center_offset_veh[mask] ** 2).sum(-1) / (sigma ** 2))
            elif ('pedestrian' in ann['category_name']):
                visibility_ped[mask] = ann['visibility_token']
                instance_ped[mask] = ins_id
                center_offset_ped[mask] = center - coords[mask]
                center_score_ped[mask] = np.exp(-(center_offset_ped[mask] ** 2).sum(-1) / (sigma ** 2))

        # update 231006
        segmentation = np.float32(segmentation[..., None])
        center_score_veh = center_score_veh[..., None]
        center_score_ped = center_score_ped[..., None]
        instance_veh = instance_veh[..., None]  
        instance_ped = instance_ped[..., None] 

        # update 231006
        result = {'segmentation': segmentation,
                    'center_ohw': center_ohw,
                    'center_score_veh': center_score_veh,
                    'center_score_ped': center_score_ped,
                    'center_offset_veh': center_offset_veh,
                    'center_offset_ped': center_offset_ped,
                    'instance_veh': instance_veh,
                    'instance_ped': instance_ped}

        return result, visibility_veh, visibility_ped, ins_map


    def get_image_visibility(self, anns, intrinsic, extrinsic, egolidar, top_crop):
        h, w = self.args.mask_h, self.args.mask_w

        visibility_veh = np.zeros((h, w), dtype=np.uint8)
        visibility_ped = np.zeros((h, w), dtype=np.uint8)
        
        # global -> ego 
        ego_translation = np.array(egolidar['translation'])
        ego_rotation = Quaternion(egolidar['rotation'])

        # visibility_token
        anns = sorted(anns, key=lambda x: x['visibility_token'])
        
        for ann in anns:
            # global -> ego
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            box.translate(-ego_translation)
            box.rotate(ego_rotation.inverse)
            
            corners_3d = np.vstack((box.corners(), np.ones((1, 8))))
            corners_cam = extrinsic @ corners_3d
            
            # 3D → 2D 
            valid_mask = corners_cam[2, :] > 0
            valid_mask = valid_mask.bool()
            if not torch.any(valid_mask):
                continue

            corners_cam = corners_cam[:3, valid_mask] 
            corners_2d = view_points(corners_cam, intrinsic, normalize=True).T[:, :2]
            corners_2d = np.clip(corners_2d, [0, 0], [w - 1, h - 1]) 

            x_min, y_min = np.min(corners_2d, axis=0).astype(int)
            x_max, y_max = np.max(corners_2d, axis=0).astype(int)

            x_min, x_max = np.clip([x_min, x_max], 0, w - 1)
            y_min, y_max = np.clip([y_min, y_max], 0, h - 1)

            if x_max > x_min and y_max > y_min:
                if 'vehicle' in ann['category_name']:
                    visibility_veh[y_min:y_max, x_min:x_max] = ann['visibility_token']
                elif 'pedestrian' in ann['category_name']:
                    visibility_ped[y_min:y_max, x_min:x_max] = ann['visibility_token']

        visibility_veh[visibility_veh == 0] = 255
        visibility_ped[visibility_ped == 0] = 255

        return visibility_veh, visibility_ped


    def traverse_linked_list(self, obj, tablekey, direction, inclusive=False):
        return nuscenes_module.traverse_linked_list(self.nusc, obj, tablekey, direction, inclusive)


    def interpolate_boxes_to_times(self, boxes, box_timestamps_relative, lidar_tokens, relative_times_interp):
        return nuscenes_module.interpolate_boxes_to_times(boxes, box_timestamps_relative, lidar_tokens, relative_times_interp)


    def get_ego_pose(self, pose_token):
        return self.nusc.get('ego_pose', pose_token)


    def transform_box(self, box, ego_pose):
        box.transform_to_pose(ego_pose)

    def return_target_bev_label(self, label, label_indices):
        target_bev = []
        for target in self.args.targets:
            for idx_group in label_indices[target]:
                # Max over specified channels in group
                target_bev.append(np.max(label[:, :, idx_group], axis=-1, keepdims=True))
        return np.concatenate(target_bev, axis=-1)
        