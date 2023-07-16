#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: baseline.py
@time: 2021/12/16 22:41
'''
import torch
import torch_scatter
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network.basic_block import Lovasz_loss
from network.base_model import LightningBaseModel
from network.basic_block import SparseBasicBlock
from network.voxel_fea_generator import voxel_3d_generator, voxelization

import logging
logging.basicConfig(format='%(pathname)s->%(lineno)d: %(message)s', level=logging.INFO)
def stop_here():
    raise RuntimeError("🚀" * 5 + "-stop-" + "🚀" * 5)


class point_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super(point_encoder, self).__init__()
        self.scale = scale
        self.layer_in = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.layer_out = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LeakyReLU(0.1, True),
            nn.Linear(out_channels, out_channels))

    @staticmethod
    def downsample(coors, p_fea, scale=2):
        batch = coors[:, 0:1]
        coors = coors[:, 1:] // scale
        inv = torch.unique(torch.cat([batch, coors], 1), return_inverse=True, dim=0)[1]   # 得到相邻体素特征的索引
        return torch_scatter.scatter_mean(p_fea, inv, dim=0), inv

    def forward(self, features, data_dict):
        logging.info("features.size: {}".format(features.shape))
        logging.info("data_dict['coors']: {}".format(data_dict['coors'].shape))
        
        output, inv = self.downsample(data_dict['coors'], features)   # 对非空体素对应的特征进行下采样
        logging.info("output: {}".format(output.shape))
        identity = self.layer_in(features)
        output = self.PPmodel(output)[inv]   # 体素数量恢复
        logging.info("identity: {}".format(identity.shape))
        logging.info("output: {}".format(output.shape))
        output = torch.cat([identity, output], dim=1)
        logging.info("output: {}".format(output.shape))

        p_feat = torch_scatter.scatter_mean(
            self.layer_out(output[data_dict['coors_inv']]),
            data_dict['scale_{}'.format(self.scale)]['coors_inv'],
            dim=0
        )
        logging.info("p_feat: {}".format(p_feat.shape))
        # stop_here()
        data_dict['coors'] = data_dict['scale_{}'.format(self.scale)]['coors']
        data_dict['coors_inv'] = data_dict['scale_{}'.format(self.scale)]['coors_inv']
        data_dict['full_coors'] = data_dict['scale_{}'.format(self.scale)]['full_coors']

        return p_feat


class SPVBlock(nn.Module):
    def __init__(self, in_channels, out_channels, indice_key, scale, last_scale, spatial_shape):
        super(SPVBlock, self).__init__()
        logging.info("in_channels: {}".format(in_channels))
        logging.info("out_channels: {}".format(out_channels))
        logging.info("indice_key: {}".format(indice_key))
        logging.info("scale: {}".format(scale))
        logging.info("last_scale: {}".format(last_scale))
        logging.info("spatial_shape: {}".format(spatial_shape))
        self.scale = scale
        self.indice_key = indice_key
        self.layer_id = indice_key.split('_')[1]
        self.last_scale = last_scale
        self.spatial_shape = spatial_shape
        self.v_enc = spconv.SparseSequential(
            SparseBasicBlock(in_channels, out_channels, self.indice_key),
            SparseBasicBlock(out_channels, out_channels, self.indice_key),
        )
        self.p_enc = point_encoder(in_channels, out_channels, scale)

    def forward(self, data_dict):
        coors_inv_last = data_dict['scale_{}'.format(self.last_scale)]['coors_inv']
        coors_inv = data_dict['scale_{}'.format(self.scale)]['coors_inv']

        # voxel encoder
        v_fea = self.v_enc(data_dict['sparse_tensor'])
        data_dict['layer_{}'.format(self.layer_id)] = {}
        data_dict['layer_{}'.format(self.layer_id)]['pts_feat'] = v_fea.features      # 点云在不同scale进行spconv后得到的特征
        data_dict['layer_{}'.format(self.layer_id)]['full_coors'] = data_dict['full_coors']  # layer_i对应
        v_fea_inv = torch_scatter.scatter_mean(v_fea.features[coors_inv_last], coors_inv, dim=0)
        logging.info("layer_{}['pts_feat'].shape: {}".format(self.layer_id, data_dict['layer_{}'.format(self.layer_id)]['pts_feat'].shape))
        logging.info("layer_{}['full_coors'].shape: {}".format(self.layer_id, data_dict['layer_{}'.format(self.layer_id)]['full_coors'].shape))

        # point encoder
        p_fea = self.p_enc(
            features=data_dict['sparse_tensor'].features+v_fea.features,
            data_dict=data_dict
        )

        # fusion and pooling
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=p_fea+v_fea_inv,
            indices=data_dict['coors'],
            spatial_shape=self.spatial_shape,
            batch_size=data_dict['batch_size']
        )

        logging.info("coors_inv: {}".format(coors_inv.shape))
        return p_fea[coors_inv]   # 返回时将每个点的特征设置为它所处scale voxel的特征

# SPVCNN
class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.input_dims = config['model_params']['input_dims']
        self.hiden_size = config['model_params']['hiden_size']
        self.num_classes = config['model_params']['num_classes']
        self.scale_list = config['model_params']['scale_list']
        self.num_scales = len(self.scale_list)
        min_volume_space = config['dataset_params']['min_volume_space']
        max_volume_space = config['dataset_params']['max_volume_space']
        self.coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                [min_volume_space[1], max_volume_space[1]],
                                [min_volume_space[2], max_volume_space[2]]]
        self.spatial_shape = np.array(config['model_params']['spatial_shape'])
        self.strides = [int(scale / self.scale_list[0]) for scale in self.scale_list]
        logging.info("self.strides: {}".format(self.strides))

        # voxelization
        self.voxelizer = voxelization(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=self.scale_list
        )

        # input processing
        self.voxel_3d_generator = voxel_3d_generator(
            in_channels=self.input_dims,
            out_channels=self.hiden_size,
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape
        )

        # encoder layers
        self.spv_enc = nn.ModuleList()
        for i in range(self.num_scales):
            self.spv_enc.append(SPVBlock(
                in_channels=self.hiden_size,
                out_channels=self.hiden_size,
                indice_key='spv_'+ str(i),
                scale=self.scale_list[i],
                last_scale=self.scale_list[i-1] if i > 0 else 1,
                spatial_shape=np.int32(self.spatial_shape // self.strides[i])[::-1].tolist())
            )
        logging.info(self.spv_enc)
        # stop_here()

        # decoder layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        # loss
        self.criterion = criterion(config)

    def forward(self, data_dict):
        with torch.no_grad():
            # for k in data_dict.keys():
            #     logging.info(k) 
            # stop_here()
            data_dict = self.voxelizer(data_dict)
            # for k in data_dict.keys():
            #     logging.info(k)
            # stop_here()

        data_dict = self.voxel_3d_generator(data_dict)
        # for k in data_dict.keys():
        #     logging.info(k)
        # stop_here()

        enc_feats = []
        for i in range(self.num_scales):
            enc_feats.append(self.spv_enc[i](data_dict))  # 不同scale对应的encoder_feature

        for k in data_dict.keys():
            logging.info(k)
        # stop_here()
        for _, iter in enumerate(enc_feats):
            logging.info("enc_feats[{}].shape: {}".format(_, iter.shape))

        output = torch.cat(enc_feats, dim=1)
        logging.info("output.shape: {}".format(output.shape))
        # stop_here()
        data_dict['logits'] = self.classifier(output)
        logging.info("data_dict['logits'].shape: {}".format(data_dict['logits'].shape))
        # stop_here()

        data_dict['loss'] = 0.
        data_dict = self.criterion(data_dict)
        # stop_here()

        return data_dict


class criterion(nn.Module):
    def __init__(self, config):
        super(criterion, self).__init__()
        self.config = config
        self.lambda_lovasz = self.config['train_params'].get('lambda_lovasz', 0.1)
        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=seg_labelweights,
            ignore_index=config['dataset_params']['ignore_label']
        )
        self.lovasz_loss = Lovasz_loss(
            ignore=config['dataset_params']['ignore_label']
        )

    def forward(self, data_dict):
        loss_main_ce = self.ce_loss(data_dict['logits'], data_dict['labels'].long())
        loss_main_lovasz = self.lovasz_loss(F.softmax(data_dict['logits'], dim=1), data_dict['labels'].long())
        loss_main = loss_main_ce + loss_main_lovasz * self.lambda_lovasz
        data_dict['loss_main_ce'] = loss_main_ce
        data_dict['loss_main_lovasz'] = loss_main_lovasz
        data_dict['loss'] += loss_main

        return data_dict