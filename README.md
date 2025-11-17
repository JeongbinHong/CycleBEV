# CycleBEV: Regularizing View Transformation Networks via View Cycle Consistency for Bird’s-Eye-View Semantic Segmentation


## Preparation
- Environments </br>

- Dataset </br>
  - Download [nuScenes](https://www.nuscenes.org/) dataset and modify the **"dataset_dir" in ./config/config.json** </br>
  - Download [pseudo annotation](https://drive.google.com/drive/folders/1ZHWtf2xI3fY5_hJBpwpYvMOaigPUSCCI) for image segmentation and move it to **./nuscenes/v1.0-trainval/** in your nuscenes path. </br>

- Pretrained weights of IVT </br>
  Download the [pretrained checkpoints](https://drive.google.com/drive/folders/10Nfm69LMlvekKCMYbMmumhpMP01OyjaS) and move it to  **./saved_models/pretrained_ck/** </br>

## Train & Inference
```
./train_cyclebev_cvt.sh
```


## Acknowledgement


## Contact
