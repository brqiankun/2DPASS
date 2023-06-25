
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
- [lightning](https://github.com/Lightning-AI/lightning) (https://lightning.ai/docs/pytorch/latest/)  (tested with pytorch_lightning==1.3.8 and torchmetrics==0.5)  `pip install pytorch_lightning==1.3.8 pip install torchmetrics==0.5`
`conda install lightning -c conda-forge` done
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

2dpass 模型结构
model_3d 是 SPVCNN
model_2d 是 ResNetFCN
fusion 是 xModalKD

data_dict -> model_3d -> model_2d -> fusion -> data_dict  ??

### data_dict
/home/bairui/program/2dpass/network/arch_2dpass.py->171: points   points[0].shape: torch.Size([119549, 4])  [xyz, sig]
/home/bairui/program/2dpass/network/arch_2dpass.py->171: ref_xyz   ref_xyz[0].shape: torch.Size([119549, 3])
/home/bairui/program/2dpass/network/arch_2dpass.py->171: batch_idx   batch_idx.shape: torch.Size([119549]) 每个点属于batch中的第几帧点云
/home/bairui/program/2dpass/network/arch_2dpass.py->171: batch_size
/home/bairui/program/2dpass/network/arch_2dpass.py->171: labels  labels.shape: torch.Size([119549, 1])
/home/bairui/program/2dpass/network/arch_2dpass.py->171: raw_labels   raw_labels.shape: (123389, 1)
/home/bairui/program/2dpass/network/arch_2dpass.py->171: origin_len   origin_len: 123389
/home/bairui/program/2dpass/network/arch_2dpass.py->171: indices
/home/bairui/program/2dpass/network/arch_2dpass.py->171: point2img_index  point2img_index[0].shape: torch.Size([8543])  一个list，batch中每帧点云可以投影到图像范围内的点云索引  
/home/bairui/program/2dpass/network/arch_2dpass.py->171: img  img[0].shape: torch.Size([320, 480, 3])
/home/bairui/program/2dpass/network/arch_2dpass.py->171: img_indices  img_indices[0].shape: (8543, 2)  batch中每帧点云投影到图像的像素坐标
/home/bairui/program/2dpass/network/arch_2dpass.py->171: img_label  img_label.shape: torch.Size([8543, 1])   根据点云.label，以及投影关系得到的投影到图像平面的点的标签
/home/bairui/program/2dpass/network/arch_2dpass.py->171: path  path: ['/home/bairui/program/2dpass/dataset/SemanticKitti/dataset/sequences/08/velodyne/000000.bin']

/home/bairui/program/2dpass/dataloader/dataset.py->182: keep_idx_img_pts: [ True  True  True ... False False False]
keep_idx [true, false ...] 包含是否符合条件的点云索引

data_dict['points'].shape: torch.Size([119549, 4])
pc.shape: torch.Size([119549, 3])
self.scale_list: [2, 4, 8, 16, 1]
self.coors_range_xyz: [[-50, 50], [-50, 50], [-4, 2]]
self.spatial_shape: [1000 1000   60]
xidx.shape: torch.Size([119549])  得到每个点在不同scale体素中的坐标


```
data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'coors_inv': unq_inv,  每个点所属的体素编号
                'coors': unq.type(torch.int32)   不同的scale对应的unq不同, scale越大， 去除重复后的bzyx
            }
```
**scale_1**
/home/bairui/program/2dpass/network/voxel_fea_generator.py->50: full_coors: bxyz_indx.shape: torch.Size([119549, 4])  对于不同scale，每个点在体素中的坐标(索引)
/home/bairui/program/2dpass/network/voxel_fea_generator.py->54: scale_1 coors_inv: unq_inv.shape: torch.Size([119549]) 每个点在不同scale对应的体素中的索引
/home/bairui/program/2dpass/network/voxel_fea_generator.py->55: scale_1 coors: unq.shape: torch.Size([64811, 4]) 不重复的体素坐标即非空的体素坐标

/home/bairui/program/2dpass/network/voxel_fea_generator.py->43: 🚀scale: 1
/home/bairui/program/2dpass/network/voxel_fea_generator.py->47: xidx.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->51: bxyz_indx.shape: torch.Size([119549, 4])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->55: scale_1 unq_inv.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->56: scale_1 unq.shape: torch.Size([64811, 4])

**scale_2**
full_coors: bxyz_indx.shape: torch.Size([119549, 4])  每个点在体素中的坐标(索引)
coors_inv: unq_inv.shape: torch.Size([119549])  每个点所属的体素编号
coors: unq.shape: torch.Size([38399, 4])   不同的scale对应的unq不同, scale越大， 去除重复后的bzyx

/home/bairui/program/2dpass/network/voxel_fea_generator.py->43: 🚀scale: 2
/home/bairui/program/2dpass/network/voxel_fea_generator.py->47: xidx.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->51: bxyz_indx.shape: torch.Size([119549, 4])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->55: scale_2 unq_inv.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->56: scale_2 coors: unq.shape: torch.Size([38399, 4])

**scale_4**
/home/bairui/program/2dpass/network/voxel_fea_generator.py->43: 🚀scale: 4
/home/bairui/program/2dpass/network/voxel_fea_generator.py->47: xidx.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->51: bxyz_indx.shape: torch.Size([119549, 4])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->55: scale_4 unq_inv.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->56: scale_4 unq.shape: torch.Size([18400, 4])

**scale_8**
/home/bairui/program/2dpass/network/voxel_fea_generator.py->43: 🚀scale: 8
/home/bairui/program/2dpass/network/voxel_fea_generator.py->47: xidx.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->51: bxyz_indx.shape: torch.Size([119549, 4])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->55: scale_8 unq_inv.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->56: scale_8 unq.shape: torch.Size([7757, 4])

**scale_16**
/home/bairui/program/2dpass/network/voxel_fea_generator.py->43: 🚀scale: 16
/home/bairui/program/2dpass/network/voxel_fea_generator.py->47: xidx.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->51: bxyz_indx.shape: torch.Size([119549, 4])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->55: scale_16 unq_inv.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->56: scale_16 unq.shape: torch.Size([2881, 4])

**scale_1**
/home/bairui/program/2dpass/network/voxel_fea_generator.py->43: 🚀scale: 1
/home/bairui/program/2dpass/network/voxel_fea_generator.py->47: xidx.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->51: bxyz_indx.shape: torch.Size([119549, 4])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->55: scale_1 unq_inv.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->56: scale_1 unq.shape: torch.Size([64811, 4])


### data_dict -> model_3d.voxelizer -> data_dict

/home/bairui/program/2dpass/network/baseline.py->174: points
/home/bairui/program/2dpass/network/baseline.py->174: ref_xyz
/home/bairui/program/2dpass/network/baseline.py->174: batch_idx
/home/bairui/program/2dpass/network/baseline.py->174: batch_size
/home/bairui/program/2dpass/network/baseline.py->174: labels
/home/bairui/program/2dpass/network/baseline.py->174: raw_labels
/home/bairui/program/2dpass/network/baseline.py->174: origin_len
/home/bairui/program/2dpass/network/baseline.py->174: indices
/home/bairui/program/2dpass/network/baseline.py->174: point2img_index  
/home/bairui/program/2dpass/network/baseline.py->174: img
/home/bairui/program/2dpass/network/baseline.py->174: img_indices
/home/bairui/program/2dpass/network/baseline.py->174: img_label
/home/bairui/program/2dpass/network/baseline.py->174: path
/home/bairui/program/2dpass/network/baseline.py->174: scale_2   不同scale对应的voxel化后每个点的坐标
/home/bairui/program/2dpass/network/baseline.py->174: scale_4
/home/bairui/program/2dpass/network/baseline.py->174: scale_8
/home/bairui/program/2dpass/network/baseline.py->174: scale_16
/home/bairui/program/2dpass/network/baseline.py->174: scale_1


### data_dict -> model.voxel_3d_generator -> data_dict

self.coors_range_xyz: [[-50, 50], [-50, 50], [-4, 2]]
self.spatial_shape: [1000 1000   60]
/home/bairui/program/2dpass/network/baseline.py->141: out_channels: hiden_size: 64
/home/bairui/program/2dpass/network/baseline.py->142: in_channels: input_dims: 4

intervals: tensor([0.1000, 0.1000, 0.1000], device='cuda:0')
grid_ind.shape: torch.Size([119549, 3])

/home/bairui/program/2dpass/network/voxel_fea_generator.py->91: point.shape: torch.Size([119549, 4])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->92: nor_pc.shape: torch.Size([119549, 3])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->93: center_to_point: torch.Size([119549, 3])
/home/bairui/program/2dpass/network/voxel_fea_generator.py->96: pc_feature.shape: torch.Size([119549, 10])  包含点云坐标和强度[0, 4], 同一体素点云的均值[5, 7], 点云与所属体素中心的偏移[8, 10]

**pt_fea = self.PPmodel(pt_fea)**
/home/bairui/program/2dpass/network/voxel_fea_generator.py->106: pt_fea.shape: torch.Size([119549, 64])

**对处于相同体素的提取后的特征求mean**
/home/bairui/program/2dpass/network/voxel_fea_generator.py->109: feature.shape: torch.Size([64811, 64])

**data_dict['sparse_tensor']**
spconv.SparseConvTensor(feature=torch.Size([64811, 64]), indices=coors: unq.shape: torch.Size([64811, 4]) (batch_idx, z, y, x), spatial_shape=[60, 1000, 1000], batch_size)

/home/bairui/program/2dpass/network/baseline.py->182: points
/home/bairui/program/2dpass/network/baseline.py->182: ref_xyz
/home/bairui/program/2dpass/network/baseline.py->182: batch_idx
/home/bairui/program/2dpass/network/baseline.py->182: batch_size
/home/bairui/program/2dpass/network/baseline.py->182: labels
/home/bairui/program/2dpass/network/baseline.py->182: raw_labels
/home/bairui/program/2dpass/network/baseline.py->182: origin_len
/home/bairui/program/2dpass/network/baseline.py->182: indices
/home/bairui/program/2dpass/network/baseline.py->182: point2img_index
/home/bairui/program/2dpass/network/baseline.py->182: img
/home/bairui/program/2dpass/network/baseline.py->182: img_indices
/home/bairui/program/2dpass/network/baseline.py->182: img_label
/home/bairui/program/2dpass/network/baseline.py->182: path
/home/bairui/program/2dpass/network/baseline.py->182: scale_2
/home/bairui/program/2dpass/network/baseline.py->182: scale_4
/home/bairui/program/2dpass/network/baseline.py->182: scale_8
/home/bairui/program/2dpass/network/baseline.py->182: scale_16
/home/bairui/program/2dpass/network/baseline.py->182: scale_1
/home/bairui/program/2dpass/network/baseline.py->182: sparse_tensor
/home/bairui/program/2dpass/network/baseline.py->182: coors       见上方scale_1[coors]
/home/bairui/program/2dpass/network/baseline.py->182: coors_inv  == ['scale_1']['coors_inv']
/home/bairui/program/2dpass/network/baseline.py->182: full_coors == ['scale_1']['full_coors']


### data_dict -> self.spv_enc(encoder: SPVBlock) -> data_dict
**spvblock**
in/out_channels = 64
indice_key = spv_0/1/2/3
scale = 2/4/8/16
last_scale = 1/2/4/8
spatial_shape: [60, 1000, 1000] / [30, 500, 500] / [15, 250, 250] / [7, 125, 125]  

/home/bairui/program/2dpass/network/baseline.py->113: layer_0['pts_feat'].shape: torch.Size([64811, 64])  不同scale的进行spconv后的点云特征
/home/bairui/program/2dpass/network/baseline.py->114: layer_0['full_coors'].shape: torch.Size([119549, 4]) point_encoder逐渐从对应scale_i中读出full_coors中，layer_i中设置为data_dict['full_coors'], 因此layer_i(0, 1, 2)对应scale_i(2, 4, 8, 16)的full_coors(即不同scale点云体素化后，每个点所在体素的坐标)

/home/bairui/program/2dpass/network/baseline.py->113: layer_1['pts_feat'].shape: torch.Size([38399, 64])
/home/bairui/program/2dpass/network/baseline.py->114: layer_1['full_coors'].shape: torch.Size([119549, 4])

/home/bairui/program/2dpass/network/baseline.py->113: layer_2['pts_feat'].shape: torch.Size([18400, 64])
/home/bairui/program/2dpass/network/baseline.py->114: layer_2['full_coors'].shape: torch.Size([119549, 4])

/home/bairui/program/2dpass/network/baseline.py->113: layer_3['pts_feat'].shape: torch.Size([7757, 64])
/home/bairui/program/2dpass/network/baseline.py->114: layer_3['full_coors'].shape: torch.Size([119549, 4])


point_encoder:
**downsample**  # 对非空体素对应的特征进行2倍下采样
/home/bairui/program/2dpass/network/baseline.py->57: features.size: torch.Size([64811, 64])
/home/bairui/program/2dpass/network/baseline.py->58: data_dict['coors']: torch.Size([64811, 4])
下采样后
/home/bairui/program/2dpass/network/baseline.py->61: output: torch.Size([38399, 64])

/home/bairui/program/2dpass/network/baseline.py->57: features.size: torch.Size([64811, 64])
/home/bairui/program/2dpass/network/baseline.py->58: data_dict['coors']: torch.Size([64811, 4])
/home/bairui/program/2dpass/network/baseline.py->61: output: torch.Size([38399, 64])
/home/bairui/program/2dpass/network/baseline.py->64: identity: torch.Size([64811, 64])
/home/bairui/program/2dpass/network/baseline.py->65: output: torch.Size([64811, 64])
/home/bairui/program/2dpass/network/baseline.py->67: output: torch.Size([64811, 128])
/home/bairui/program/2dpass/network/baseline.py->74: v_feat: torch.Size([38399, 64])


/home/bairui/program/2dpass/network/baseline.py->192: points
/home/bairui/program/2dpass/network/baseline.py->192: ref_xyz
/home/bairui/program/2dpass/network/baseline.py->192: batch_idx
/home/bairui/program/2dpass/network/baseline.py->192: batch_size
/home/bairui/program/2dpass/network/baseline.py->192: labels
/home/bairui/program/2dpass/network/baseline.py->192: raw_labels
/home/bairui/program/2dpass/network/baseline.py->192: origin_len
/home/bairui/program/2dpass/network/baseline.py->192: indices
/home/bairui/program/2dpass/network/baseline.py->192: point2img_index
/home/bairui/program/2dpass/network/baseline.py->192: img
/home/bairui/program/2dpass/network/baseline.py->192: img_indices
/home/bairui/program/2dpass/network/baseline.py->192: img_label
/home/bairui/program/2dpass/network/baseline.py->192: path
/home/bairui/program/2dpass/network/baseline.py->192: scale_2
/home/bairui/program/2dpass/network/baseline.py->192: scale_4
/home/bairui/program/2dpass/network/baseline.py->192: scale_8
/home/bairui/program/2dpass/network/baseline.py->192: scale_16
/home/bairui/program/2dpass/network/baseline.py->192: scale_1
/home/bairui/program/2dpass/network/baseline.py->192: sparse_tensor
/home/bairui/program/2dpass/network/baseline.py->192: coors
/home/bairui/program/2dpass/network/baseline.py->192: coors_inv
/home/bairui/program/2dpass/network/baseline.py->192: full_coors
/home/bairui/program/2dpass/network/baseline.py->192: layer_0
/home/bairui/program/2dpass/network/baseline.py->192: layer_1
/home/bairui/program/2dpass/network/baseline.py->192: layer_2
/home/bairui/program/2dpass/network/baseline.py->192: layer_3

### data_dict -> model_3d -> data_dict
/home/bairui/program/2dpass/network/arch_2dpass.py->172: points
/home/bairui/program/2dpass/network/arch_2dpass.py->172: ref_xyz
/home/bairui/program/2dpass/network/arch_2dpass.py->172: batch_idx
/home/bairui/program/2dpass/network/arch_2dpass.py->172: batch_size
/home/bairui/program/2dpass/network/arch_2dpass.py->172: labels
/home/bairui/program/2dpass/network/arch_2dpass.py->172: raw_labels
/home/bairui/program/2dpass/network/arch_2dpass.py->172: origin_len
/home/bairui/program/2dpass/network/arch_2dpass.py->172: indices
/home/bairui/program/2dpass/network/arch_2dpass.py->172: point2img_index
/home/bairui/program/2dpass/network/arch_2dpass.py->172: img
/home/bairui/program/2dpass/network/arch_2dpass.py->172: img_indices
/home/bairui/program/2dpass/network/arch_2dpass.py->172: img_label
/home/bairui/program/2dpass/network/arch_2dpass.py->172: path
/home/bairui/program/2dpass/network/arch_2dpass.py->172: scale_2
/home/bairui/program/2dpass/network/arch_2dpass.py->172: scale_4
/home/bairui/program/2dpass/network/arch_2dpass.py->172: scale_8
/home/bairui/program/2dpass/network/arch_2dpass.py->172: scale_16
/home/bairui/program/2dpass/network/arch_2dpass.py->172: scale_1
/home/bairui/program/2dpass/network/arch_2dpass.py->172: sparse_tensor
/home/bairui/program/2dpass/network/arch_2dpass.py->172: coors
/home/bairui/program/2dpass/network/arch_2dpass.py->172: coors_inv
/home/bairui/program/2dpass/network/arch_2dpass.py->172: full_coors
/home/bairui/program/2dpass/network/arch_2dpass.py->172: layer_0
/home/bairui/program/2dpass/network/arch_2dpass.py->172: layer_1
/home/bairui/program/2dpass/network/arch_2dpass.py->172: layer_2
/home/bairui/program/2dpass/network/arch_2dpass.py->172: layer_3
/home/bairui/program/2dpass/network/arch_2dpass.py->172: logits
/home/bairui/program/2dpass/network/arch_2dpass.py->172: loss
/home/bairui/program/2dpass/network/arch_2dpass.py->172: loss_main_ce
/home/bairui/program/2dpass/network/arch_2dpass.py->172: loss_main_lovasz

不同scale的特征提取后对应每个点的特征
/home/bairui/program/2dpass/network/baseline.py->212: enc_feats[0].shape: torch.Size([119549, 64])
/home/bairui/program/2dpass/network/baseline.py->212: enc_feats[1].shape: torch.Size([119549, 64])
/home/bairui/program/2dpass/network/baseline.py->212: enc_feats[2].shape: torch.Size([119549, 64])
/home/bairui/program/2dpass/network/baseline.py->212: enc_feats[3].shape: torch.Size([119549, 64])
/home/bairui/program/2dpass/network/baseline.py->215: output.shape: torch.Size([119549, 256])
output.shape: torch.Size([119549, 256])
256 = hiden_size * num_class
**得到logits**
data_dict['logits'] = self.classifier(output)
/home/bairui/program/2dpass/network/baseline.py->218: data_dict['logits'].shape: torch.Size([119549, 20])

**得到loss**
ce_loss: logits <-> labels
lovasz_loss: softmax(logits) <-> labels


### data_dict -> model_2d -> data_dict
x.shape:torch.Size([1, 3, 320, 480])
conv1_out.shape:torch.Size([1, 64, 320, 480])
**encoder**
/home/bairui/program/2dpass/network/basic_block.py->103: layer1_out.shape:torch.Size([1, 64, 160, 240])
/home/bairui/program/2dpass/network/basic_block.py->104: layer2_out.shape:torch.Size([1, 128, 80, 120])
/home/bairui/program/2dpass/network/basic_block.py->105: layer3_out.shape:torch.Size([1, 256, 40, 60])
/home/bairui/program/2dpass/network/basic_block.py->106: layer4_out.shape:torch.Size([1, 512, 20, 30])

**deconv**
/home/bairui/program/2dpass/network/basic_block.py->116: layer1_out.shape:torch.Size([1, 64, 320, 480])
/home/bairui/program/2dpass/network/basic_block.py->117: layer2_out.shape:torch.Size([1, 64, 320, 480])
/home/bairui/program/2dpass/network/basic_block.py->118: layer3_out.shape:torch.Size([1, 64, 320, 480])
/home/bairui/program/2dpass/network/basic_block.py->119: layer4_out.shape:torch.Size([1, 64, 320, 480])

/home/bairui/program/2dpass/network/basic_block.py->129: process_keys: ['img_scale2', 'img_scale4', 'img_scale8', 'img_scale16']
/home/bairui/program/2dpass/network/basic_block.py->130: img_indices[0]: (8543, 2)
##### 图像进行特征提取并进行反卷积(上采样)后的特征
/home/bairui/program/2dpass/network/basic_block.py->132: data_dict[img_scale2].shape: torch.Size([1, 64, 320, 480])
/home/bairui/program/2dpass/network/basic_block.py->132: data_dict[img_scale4].shape: torch.Size([1, 64, 320, 480])
/home/bairui/program/2dpass/network/basic_block.py->132: data_dict[img_scale8].shape: torch.Size([1, 64, 320, 480])
/home/bairui/program/2dpass/network/basic_block.py->132: data_dict[img_scale16].shape: torch.Size([1, 64, 320, 480])
##### 筛选出可以和点云投影对应的图像特征
/home/bairui/program/2dpass/network/basic_block.py->138: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/basic_block.py->138: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/basic_block.py->138: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/basic_block.py->138: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/basic_block.py->143: data_dict[img_scale2].shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/basic_block.py->143: data_dict[img_scale4].shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/basic_block.py->143: data_dict[img_scale8].shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/basic_block.py->143: data_dict[img_scale16].shape: torch.Size([8543, 64])

/home/bairui/program/2dpass/network/arch_2dpass.py->179: points
/home/bairui/program/2dpass/network/arch_2dpass.py->179: ref_xyz
/home/bairui/program/2dpass/network/arch_2dpass.py->179: batch_idx
/home/bairui/program/2dpass/network/arch_2dpass.py->179: batch_size
/home/bairui/program/2dpass/network/arch_2dpass.py->179: labels
/home/bairui/program/2dpass/network/arch_2dpass.py->179: raw_labels
/home/bairui/program/2dpass/network/arch_2dpass.py->179: origin_len
/home/bairui/program/2dpass/network/arch_2dpass.py->179: indices
/home/bairui/program/2dpass/network/arch_2dpass.py->179: point2img_index
/home/bairui/program/2dpass/network/arch_2dpass.py->179: img
/home/bairui/program/2dpass/network/arch_2dpass.py->179: img_indices
/home/bairui/program/2dpass/network/arch_2dpass.py->179: img_label
/home/bairui/program/2dpass/network/arch_2dpass.py->179: path
/home/bairui/program/2dpass/network/arch_2dpass.py->179: scale_2
/home/bairui/program/2dpass/network/arch_2dpass.py->179: scale_4
/home/bairui/program/2dpass/network/arch_2dpass.py->179: scale_8
/home/bairui/program/2dpass/network/arch_2dpass.py->179: scale_16
/home/bairui/program/2dpass/network/arch_2dpass.py->179: scale_1
/home/bairui/program/2dpass/network/arch_2dpass.py->179: sparse_tensor
/home/bairui/program/2dpass/network/arch_2dpass.py->179: coors
/home/bairui/program/2dpass/network/arch_2dpass.py->179: coors_inv
/home/bairui/program/2dpass/network/arch_2dpass.py->179: full_coors
/home/bairui/program/2dpass/network/arch_2dpass.py->179: layer_0
/home/bairui/program/2dpass/network/arch_2dpass.py->179: layer_1
/home/bairui/program/2dpass/network/arch_2dpass.py->179: layer_2
/home/bairui/program/2dpass/network/arch_2dpass.py->179: layer_3
/home/bairui/program/2dpass/network/arch_2dpass.py->179: logits
/home/bairui/program/2dpass/network/arch_2dpass.py->179: loss
/home/bairui/program/2dpass/network/arch_2dpass.py->179: loss_main_ce
/home/bairui/program/2dpass/network/arch_2dpass.py->179: loss_main_lovasz
/home/bairui/program/2dpass/network/arch_2dpass.py->179: img_scale2          torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->179: img_scale4          torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->179: img_scale8          torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->179: img_scale16         torch.Size([8543, 64])


### data_dict -> fusion -> data_dict
labels.shape: torch.Size([119549, 1])

/home/bairui/program/2dpass/network/arch_2dpass.py->104: img_scale2 img_feat.shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->105: layer_0 pts_feat.shape: torch.Size([64811, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->106: scale_1 corrs_inv.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/arch_2dpass.py->110: pts_pred_full.shape: torch.Size([64811, 20])
/home/bairui/program/2dpass/network/arch_2dpass.py->81: lbxyz.shape: torch.Size([119549, 5])
/home/bairui/program/2dpass/network/arch_2dpass.py->82: unq_lbxyz.shape: torch.Size([65119, 5])
/home/bairui/program/2dpass/network/arch_2dpass.py->83: count.shape: torch.Size([65119])
/home/bairui/program/2dpass/network/arch_2dpass.py->84: inv_ind.shape: torch.Size([65119])
/home/bairui/program/2dpass/network/arch_2dpass.py->85: label_ind.shape: torch.Size([64811])
/home/bairui/program/2dpass/network/arch_2dpass.py->87: labels.shape: torch.Size([64811])

**p2img_mapping**
/home/bairui/program/2dpass/network/arch_2dpass.py->72: pts_fea.shape: torch.Size([119549, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->73: p2img_idx[0].shape: torch.Size([8543])
/home/bairui/program/2dpass/network/arch_2dpass.py->74: batch_idx.shape: torch.Size([119549])

/home/bairui/program/2dpass/network/arch_2dpass.py->72: pts_fea.shape: torch.Size([119549, 20])
/home/bairui/program/2dpass/network/arch_2dpass.py->73: p2img_idx[0].shape: torch.Size([8543])
/home/bairui/program/2dpass/network/arch_2dpass.py->74: batch_idx.shape: torch.Size([119549])

/home/bairui/program/2dpass/network/arch_2dpass.py->97: img_scale4 img_feat.shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->98: layer_1 pts_feat.shape: torch.Size([38399, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->99: scale_2 corrs_inv.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/arch_2dpass.py->103: pts_pred_full.shape: torch.Size([38399, 20])
/home/bairui/program/2dpass/network/arch_2dpass.py->140: fuse_feat.shape: torch.Size([8543, 64])

/home/bairui/program/2dpass/network/arch_2dpass.py->97: img_scale8 img_feat.shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->98: layer_2 pts_feat.shape: torch.Size([18400, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->99: scale_4 corrs_inv.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/arch_2dpass.py->103: pts_pred_full.shape: torch.Size([18400, 20])
/home/bairui/program/2dpass/network/arch_2dpass.py->140: fuse_feat.shape: torch.Size([8543, 64])

/home/bairui/program/2dpass/network/arch_2dpass.py->97: img_scale16 img_feat.shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->98: layer_3 pts_feat.shape: torch.Size([7757, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->99: scale_8 corrs_inv.shape: torch.Size([119549])
/home/bairui/program/2dpass/network/arch_2dpass.py->103: pts_pred_full.shape: torch.Size([7757, 20])
/home/bairui/program/2dpass/network/arch_2dpass.py->140: fuse_feat.shape: torch.Size([8543, 64])

/home/bairui/program/2dpass/network/arch_2dpass.py->118: pts_feat.shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->120: pts_pred.shape: torch.Size([8543, 20])

/home/bairui/program/2dpass/network/arch_2dpass.py->124: feat_learner.shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->126: feat_cat.shape: torch.Size([8543, 128])
**经过self.fcs1/self.fcs2**后得到融合特征
/home/bairui/program/2dpass/network/arch_2dpass.py->130: fuse_feat.shape: torch.Size([8543, 64])

融合预测结果
/home/bairui/program/2dpass/network/arch_2dpass.py->134: fuse_pred.shape: torch.Size([8543, 20])

/home/bairui/program/2dpass/network/arch_2dpass.py->161: scale_2: fuse_feat.shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->161: scale_4: fuse_feat.shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->161: scale_8: fuse_feat.shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->161: scale_16: fuse_feat.shape: torch.Size([8543, 64])
/home/bairui/program/2dpass/network/arch_2dpass.py->165: torch.cat(img_seg_feat, 1).shape: torch.Size([8543, 256])
/home/bairui/program/2dpass/network/arch_2dpass.py->167: img_seg_logits.shape: torch.Size([8543, 20])




/home/bairui/program/2dpass/network/arch_2dpass.py->183: points
/home/bairui/program/2dpass/network/arch_2dpass.py->183: ref_xyz
/home/bairui/program/2dpass/network/arch_2dpass.py->183: batch_idx
/home/bairui/program/2dpass/network/arch_2dpass.py->183: batch_size
/home/bairui/program/2dpass/network/arch_2dpass.py->183: labels
/home/bairui/program/2dpass/network/arch_2dpass.py->183: raw_labels
/home/bairui/program/2dpass/network/arch_2dpass.py->183: origin_len
/home/bairui/program/2dpass/network/arch_2dpass.py->183: indices
/home/bairui/program/2dpass/network/arch_2dpass.py->183: point2img_index
/home/bairui/program/2dpass/network/arch_2dpass.py->183: img
/home/bairui/program/2dpass/network/arch_2dpass.py->183: img_indices
/home/bairui/program/2dpass/network/arch_2dpass.py->183: img_label
/home/bairui/program/2dpass/network/arch_2dpass.py->183: path
/home/bairui/program/2dpass/network/arch_2dpass.py->183: scale_2
/home/bairui/program/2dpass/network/arch_2dpass.py->183: scale_4
/home/bairui/program/2dpass/network/arch_2dpass.py->183: scale_8
/home/bairui/program/2dpass/network/arch_2dpass.py->183: scale_16
/home/bairui/program/2dpass/network/arch_2dpass.py->183: scale_1
/home/bairui/program/2dpass/network/arch_2dpass.py->183: sparse_tensor
/home/bairui/program/2dpass/network/arch_2dpass.py->183: coors
/home/bairui/program/2dpass/network/arch_2dpass.py->183: coors_inv
/home/bairui/program/2dpass/network/arch_2dpass.py->183: full_coors
/home/bairui/program/2dpass/network/arch_2dpass.py->183: layer_0
/home/bairui/program/2dpass/network/arch_2dpass.py->183: layer_1
/home/bairui/program/2dpass/network/arch_2dpass.py->183: layer_2
/home/bairui/program/2dpass/network/arch_2dpass.py->183: layer_3
/home/bairui/program/2dpass/network/arch_2dpass.py->183: logits
/home/bairui/program/2dpass/network/arch_2dpass.py->183: loss
/home/bairui/program/2dpass/network/arch_2dpass.py->183: loss_main_ce
/home/bairui/program/2dpass/network/arch_2dpass.py->183: loss_main_lovasz
/home/bairui/program/2dpass/network/arch_2dpass.py->183: img_scale2
/home/bairui/program/2dpass/network/arch_2dpass.py->183: img_scale4
/home/bairui/program/2dpass/network/arch_2dpass.py->183: img_scale8
/home/bairui/program/2dpass/network/arch_2dpass.py->183: img_scale16

### spvnas
Sparse Point-Voxel Convolution

基于点的方法point-based methods在处理非结构化数据上耗费超过90%时间
基于体素的方法在体素分辨率的选取收到很大影响
提出Sparse Point-Voxel Convolution稀疏点体素卷积

1. point-based branch 保持高分辨率
2. sparse voxel-based branch 使用稀疏卷积来跨不同的感受野
两个分支通过稀疏体素化和反体素化来进行结合

使用GPU hash table来加速稀疏voxel化和反voxel
spconv



