
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/2dpass-2d-priors-assisted-semantic/3d-semantic-segmentation-on-semantickitti)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-semantickitti?p=2dpass-2d-priors-assisted-semantic)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/2dpass-2d-priors-assisted-semantic/lidar-semantic-segmentation-on-nuscenes)](https://paperswithcode.com/sota/lidar-semantic-segmentation-on-nuscenes?p=2dpass-2d-priors-assisted-semantic)

# 2DPASS

[![arXiv](https://img.shields.io/badge/arXiv-2203.09065-b31b1b.svg)](https://arxiv.org/pdf/2207.04397.pdf)
[![GitHub Stars](https://img.shields.io/github/stars/yanx27/2DPASS?style=social)](https://github.com/yanx27/2DPASS)
![visitors](https://visitor-badge.glitch.me/badge?page_id=https://github.com/yanx27/2DPASS)



This repository is for **2DPASS** introduced in the following paper

[Xu Yan*](https://yanx27.github.io/), [Jiantao Gao*](https://github.com/Gao-JT), [Chaoda Zheng*](https://github.com/Ghostish), Chao Zheng, Ruimao Zhang, Shuguang Cui, [Zhen Li*](https://mypage.cuhk.edu.cn/academics/lizhen/), "*2DPASS: 2D Priors Assisted Semantic Segmentation on LiDAR Point Clouds*", ECCV 2022 [[arxiv]](https://arxiv.org/pdf/2207.04397.pdf).
 ![image](figures/2DPASS.gif)

If you find our work useful in your research, please consider citing:
```latex
@inproceedings{yan20222dpass,
  title={2dpass: 2d priors assisted semantic segmentation on lidar point clouds},
  author={Yan, Xu and Gao, Jiantao and Zheng, Chaoda and Zheng, Chao and Zhang, Ruimao and Cui, Shuguang and Li, Zhen},
  booktitle={European Conference on Computer Vision},
  pages={677--695},
  year={2022},
  organization={Springer}
}

@InProceedings{yan2022let,
      title={Let Images Give You More: Point Cloud Cross-Modal Training for Shape Analysis}, 
      author={Xu Yan and Heshen Zhan and Chaoda Zheng and Jiantao Gao and Ruimao Zhang and Shuguang Cui and Zhen Li},
      year={2022},
      booktitle={NeurIPS}
}

@article{yan2023benchmarking,
  title={Benchmarking the Robustness of LiDAR Semantic Segmentation Models},
  author={Yan, Xu and Zheng, Chaoda and Li, Zhen and Cui, Shuguang and Dai, Dengxin},
  journal={arXiv preprint arXiv:2301.00970},
  year={2023}
}
```
## News
* **2023-04-01** We merge MinkowskiNet and official SPVCNN models from [SPVNAS](https://github.com/mit-han-lab/spvnas) in our codebase. You can check these models in `config/`. We rename our baseline model from `spvcnn.py` to `baseline.py`.
* **2023-03-31** We provide codes for the robustness evaluation on SemanticKITTI-C.
* **2023-03-27** We release a model with higher performance on SemanticKITTI and codes for naive instance augmentation.
* **2023-02-25** We release a new robustness benchmark for LiDAR semantic segmentation at [SemanticKITTI-C](https://yanx27.github.io/RobustLidarSeg/). Welcome to test your models!
<p align="center">
   <img src="figures/semantickittic.png" width="80%"> 
</p>


* **2022-10-11** Our new work for cross-modal knowledge distillation is accepted at NeurIPS 2022:smiley: [paper](https://arxiv.org/pdf/2210.04208.pdf) / [code](https://github.com/ZhanHeshen/PointCMT).
* **2022-09-20** We release codes for SemanticKITTI single-scan and NuScenes :rocket:!
* **2022-07-03** 2DPASS is accepted at **ECCV 2022** :fire:!
* **2022-03-08** We achieve **1st** place in both single and multi-scans of [SemanticKITTI](http://semantic-kitti.org/index.html) and **3rd** place on [NuScenes-lidarseg](https://www.nuscenes.org/) :fire:! 
<p align="center">
   <img src="figures/singlescan.jpg" width="80%"> 
</p>
<p align="center">
   <img src="figures/multiscan.jpg" width="80%"> 
</p>
<p align="center">
   <img src="figures/nuscene.png" width="80%"> 
</p>

## Installation

### Requirements
- pytorch >= 1.8 
- yaml
- easydict  `conda install -c conda-forge easydict` done
- pyquaternion   四元数库 `conda install -c conda-forge quaternion`  this `pip install pyquaternion` (http://kieranwynn.github.io/pyquaternion/)  done
- [lightning](https://github.com/Lightning-AI/lightning) (https://lightning.ai/docs/pytorch/latest/)  (tested with pytorch_lightning==1.3.8 and torchmetrics==0.5)  `conda install lightning -c conda-forge` done
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter) (pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html) done
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) `pip install nuscenes-devkit` done (optional for nuScenes)
- [spconv](https://github.com/traveller59/spconv) (tested with spconv==2.1.16 and cuda==11.1, pip install spconv-cu111==2.1.16) done 
- [torchsparse](https://github.com/mit-han-lab/torchsparse) (optional for MinkowskiNet and SPVCNN. sudo apt-get install libsparsehash-dev, pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0)
- pip install -U tensorboard
- pip install -U tensorboardX

## Data Preparation

### SemanticKITTI
Please download the files from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html) and additionally the [color data](http://www.cvlibs.net/download.php?file=data_odometry_color.zip) from the [Kitti Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extract everything into the same folder.
```
./dataset/
├── 
├── ...
└── SemanticKitti/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |   |   ├── 000000.label
        |   |   ├── 000001.label
        |   |   └── ...
        |   └── image_2/ 
        |   |   ├── 000000.png
        |   |   ├── 000001.png
        |   |   └── ...
        |   calib.txt
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### NuScenes
Please download the Full dataset (v1.0) from the [NuScenes website](https://www.nuscenes.org/) with lidarseg and extract it.
```
./dataset/
├── 
├── ...
└── nuscenes/
    ├──v1.0-trainval
    ├──v1.0-test
    ├──samples
    ├──sweeps
    ├──maps
    ├──lidarseg
```

## Training
### SemanticKITTI
You can run the training with
batch_size 设置为2可以训练， 1个epoch 2:40:00  
```shell script
cd <root dir of this repo>
python main.py --log_dir 2DPASS_semkitti --config config/2DPASS-semantickitti.yaml --gpu 0
```
The output will be written to `logs/SemanticKITTI/2DPASS_semkitti` by default. 
### NuScenes
```shell script
cd <root dir of this repo>
python main.py --log_dir 2DPASS_nusc --config config/2DPASS-nuscenese.yaml --gpu 0 1 2 3
```

### Vanilla Training without 2DPASS
We take SemanticKITTI as an example.
```shell script
cd <root dir of this repo>
python main.py --log_dir baseline_semkitti --config config/2DPASS-semantickitti.yaml --gpu 0 --baseline_only
```

## Testing
You can run the testing with
```shell script
cd <root dir of this repo>
python main.py --config config/2DPASS-semantickitti.yaml --gpu 0 --test --num_vote 12 --checkpoint <dir for the pytorch checkpoint>

python main.py --config checkpoint/2DPASS-semantickitti.yaml --gpu 0 --test --num_vote 1 --checkpoint checkpoint/best_model.ckpt
```
Here, `num_vote` is the number of views for the test-time-augmentation (TTA). We set this value to 12 as default (on a Tesla-V100 GPU), and if you use other GPUs with smaller memory, you can choose a smaller value. `num_vote=1` denotes there is no TTA used, and will cause about ~2\% performance drop.

## Robustness Evaluation
Please download all subsets of [SemanticKITTI-C](https://arxiv.org/pdf/2301.00970.pdf) from [this link](https://cuhko365-my.sharepoint.com/personal/218012048_link_cuhk_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F218012048%5Flink%5Fcuhk%5Fedu%5Fcn%2FDocuments%2FSemanticKITTIC&ga=1) and extract them.
```
./dataset/
├── 
├── ...
└── SemanticKitti/
    ├──sequences
    ├──SemanticKITTI-C
        ├── clean_data/           
        ├── dense_16beam/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |   |   ├── 000000.label
        |   |   ├── 000001.label
        |   |   └── ...
	    ...
```
You can run the robustness evaluation with
```shell script
cd <root dir of this repo>
python robust_test.py --config config/2DPASS-semantickitti.yaml --gpu 0  --num_vote 12 --checkpoint <dir for the pytorch checkpoint>
```

## Model Zoo
You can download the models with the scores below from [this Google drive folder](https://drive.google.com/drive/folders/1Xy6p_h827lv8J-2iZU8T6SLFkxfoXPBE?usp=sharing).
### SemanticKITTI
|Model (validation)|mIoU (vanilla)|mIoU (TTA)|Parameters|
|:---:|:---:|:---:|:---:|
|MinkowskiNet|65.1%|67.1%|21.7M|
|SPVCNN|65.9%|67.8%|21.8M|
|2DPASS (4scale-64dimension)|68.7%|70.0%|1.9M|
|2DPASS (6scale-256dimension)|70.7%|72.0%|45.6M|

Here, we fine-tune 2DPASS models on SemanticKITTI with more epochs and thus gain the higher mIoU. If you train with 64 epochs, it should be gained about 66%/69% for vanilla and 69%/71% after TTA.

### NuScenes
|Model (validation)|mIoU (vanilla)|mIoU (TTA)|Parameters|
|:---:|:---:|:---:|:---:|
|MinkowskiNet|74.3%|76.0%|21.7M|
|SPVCNN|74.9%|76.9%|21.8M|
|2DPASS (6scale-128dimension)|76.7%|79.6%|11.5M|
|2DPASS (6scale-256dimension)|78.0%|80.5%|45.6M|

**Note that the results on benchmarks are gained by training with additional validation set and using instance-level augmentation.**

## Acknowledgements
Code is built based on [SPVNAS](https://github.com/mit-han-lab/spvnas), [Cylinder3D](https://github.com/xinge008/Cylinder3D), [xMUDA](https://github.com/valeoai/xmuda) and [SPCONV](https://github.com/traveller59/spconv).

## License
This repository is released under MIT License (see LICENSE file for details).


因此，在这项工作中，我们提出了 2D 先验辅助语义分割（2DPASS）方法，一种通用训练方案，用于促进点云上的表示学习。所提出的2DPAS方法充分利用了训练过程中丰富的2D图像，然后在没有严格配对数据约束的情况下进行语义分割。
在实践中，通过利用辅助模态融合和多尺度融合到单一知识蒸馏 （MSFSKD），2DPASS 从多模态数据中获取更丰富的语义和结构信息，然后将其提炼到纯 3D 网络中。因此，我们的基线模型在配备 2DPASS 后仅使用点云输入即可获得显着改进。

```
please install torchsparse if you want to run spvcnn/minkowskinet!
{'format_version': 1, 'model_params': {'model_architecture': 'arch_2dpass', 'input_dims': 4, 'spatial_shape': [1000, 1000, 60], 'scale_list': [2, 4, 8, 16], 'hiden_size': 64, 'num_classes': 20, 'backbone_2d': 'resnet34', 'pretrained2d': False}, 'dataset_params': {'training_size': 19132, 'dataset_type': 'point_image_dataset_semkitti', 'pc_dataset_type': 'SemanticKITTI', 'collate_type': 'collate_fn_default', 'ignore_label': 0, 'label_mapping': './config/label_mapping/semantic-kitti.yaml', 'bottom_crop': [480, 320], 'color_jitter': [0.4, 0.4, 0.4], 'flip2d': 0.5, 'image_normalizer': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]], 'max_volume_space': [50, 50, 2], 'min_volume_space': [-50, -50, -4], 'seg_labelweights': [0, 55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858, 240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114, 9833174, 129609852, 4506626, 1168181], 'train_data_loader': {'data_path': './dataset/SemanticKitti/dataset/sequences/', 'batch_size': 8, 'shuffle': True, 'num_workers': 8, 'rotate_aug': True, 'flip_aug': True, 'scale_aug': True, 'transform_aug': True, 'dropout_aug': True}, 'val_data_loader': {'data_path': './dataset/SemanticKitti/dataset/sequences/', 'shuffle': False, 'num_workers': 8, 'batch_size': 1, 'rotate_aug': False, 'flip_aug': False, 'scale_aug': False, 'transform_aug': False, 'dropout_aug': False}}, 'train_params': {'max_num_epochs': 64, 'learning_rate': 0.24, 'optimizer': 'SGD', 'lr_scheduler': 'CosineAnnealingWarmRestarts', 'momentum': 0.9, 'nesterov': True, 'weight_decay': 0.0001, 'lambda_seg2d': 1, 'lambda_xm': 0.05}, 'gpu': [0], 'seed': 0, 'config_path': 'checkpoint/2DPASS-semantickitti.yaml', 'log_dir': 'default', 'monitor': 'val/mIoU', 'stop_patience': 50, 'save_top_k': 1, 'check_val_every_n_epoch': 1, 'SWA': False, 'baseline_only': False, 'test': True, 'fine_tune': False, 'pretrain2d': False, 'num_vote': 1, 'submit_to_server': False, 'checkpoint': 'checkpoint/best_model.ckpt', 'debug': False}
Global seed set to 0
load pre-trained model...
Start testing...
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
Global seed set to 0
initializing ddp: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All DDP processes registered. Starting ddp with 1 processes
----------------------------------------------------------------------------------------------------

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Validation per class iou:                                                                                                                                                                                                             
car : 96.83%
bicycle : 52.55%
motorcycle : 76.33%
truck : 90.74%
bus : 71.38%
person : 78.36%
bicyclist : 92.35%
motorcyclist : 0.06%
road : 93.24%
parking : 50.75%
sidewalk : 80.08%
other-ground : 8.44%
building : 92.21%
fence : 68.27%
vegetation : 88.37%
trunk : 71.19%
terrain : 74.63%
pole : 63.92%
traffic-sign : 53.46%
Current val miou is 68.587 while the best val miou is 68.587
Testing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4071/4071 [06:19<00:00, 10.72it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'val/acc': 0.8935943841934204,
 'val/best_miou': 0.6858724848558865,
 'val/mIoU': 0.6858724848558865}
--------------------------------------------------------------------------------
```