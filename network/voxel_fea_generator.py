#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: voxel_fea_generator.py
@time: 2021/8/4 13:36
'''
import torch
import torch_scatter
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv


import logging
logging.basicConfig(format='%(pathname)s->%(lineno)d: %(message)s', level=logging.INFO)
def stop_here():
    raise RuntimeError("🚀" * 5 + "-stop-" + "🚀" * 5)


class voxelization(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape, scale_list):
        super(voxelization, self).__init__()
        self.spatial_shape = spatial_shape
        self.scale_list = scale_list + [1]   # add scale 1
        self.coors_range_xyz = coors_range_xyz

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]
        logging.info("🚀 pc.shape: {}".format(pc.shape))
        logging.info("🚀 data_dict['points'].shape: {}".format(data_dict['points'].shape))
        logging.info("self.scale_list: {}".format(self.scale_list))
        logging.info("self.coors_range_xyz: {}".format(self.coors_range_xyz))
        logging.info("self.spatial_shape: {}".format(self.spatial_shape))

        # 得到每个点云在体素中的坐标
        for idx, scale in enumerate(self.scale_list):
            logging.info("🚀scale: {}".format(scale))
            xidx = self.sparse_quantize(pc[:, 0], self.coors_range_xyz[0], np.ceil(self.spatial_shape[0] / scale))  # scale相当于将点按照比例体素化到不同大小的网格中
            yidx = self.sparse_quantize(pc[:, 1], self.coors_range_xyz[1], np.ceil(self.spatial_shape[1] / scale))
            zidx = self.sparse_quantize(pc[:, 2], self.coors_range_xyz[2], np.ceil(self.spatial_shape[2] / scale))
            logging.info("xidx.shape: {}".format(xidx.shape))
            # stop_here()

            bxyz_indx = torch.stack([data_dict['batch_idx'], xidx, yidx, zidx], dim=-1).long()
            logging.info("bxyz_indx.shape: {}".format(bxyz_indx.shape))
            # stop_here()
            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)  # 位于相同体素中的坐标只保留一个
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)   # 交换x, y, z的顺序
            logging.info("scale_{} unq_inv.shape: {}".format(scale, unq_inv.shape))
            logging.info("scale_{} unq.shape: {}".format(scale, unq.shape))
            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'coors_inv': unq_inv,
                'coors': unq.type(torch.int32)
            }
        return data_dict


class voxel_3d_generator(nn.Module):
    def __init__(self, in_channels, out_channels, coors_range_xyz, spatial_shape):
        super(voxel_3d_generator, self).__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels + 6, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels)
        )

    def prepare_input(self, point, grid_ind, inv_idx):
        logging.info(point.shape)
        logging.info(inv_idx.shape)
        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx]  #  torch.Size([119549, 3]) 位于相同体素内点的坐标均值
        logging.info(pc_mean.shape)
        # stop_here()
        nor_pc = point[:, :3] - pc_mean

        coors_range_xyz = torch.Tensor(self.coors_range_xyz)
        cur_grid_size = torch.Tensor(self.spatial_shape)
        crop_range = coors_range_xyz[:, 1] - coors_range_xyz[:, 0]
        intervals = (crop_range / cur_grid_size).to(point.device)
        logging.info("intervals: {}".format(intervals))
        logging.info("grid_ind.shape: {}".format(grid_ind.shape))
        voxel_centers = grid_ind * intervals + coors_range_xyz[:, 0].to(point.device)
        center_to_point = point[:, :3] - voxel_centers   # 每个点对于体素中心的偏移
        logging.info("point.shape: {}".format(point.shape))
        logging.info("nor_pc.shape: {}".format(nor_pc.shape))
        logging.info("center_to_point: {}".format(center_to_point.shape))

        pc_feature = torch.cat((point, nor_pc, center_to_point), dim=1)
        logging.info("pc_feature.shape: {}".format(pc_feature.shape))
        return pc_feature

    def forward(self, data_dict):
        pt_fea = self.prepare_input(
            data_dict['points'],
            data_dict['scale_1']['full_coors'][:, 1:],
            data_dict['scale_1']['coors_inv']
        )
        pt_fea = self.PPmodel(pt_fea)
        logging.info("pt_fea.shape: {}".format(pt_fea.shape))

        features = torch_scatter.scatter_mean(pt_fea, data_dict['scale_1']['coors_inv'], dim=0)  # 对处于相同体素的提取后的特征求mean
        logging.info("feature.shape: {}".format(features.shape))
        logging.info("np.int32(self.spatial_shape)[::-1].tolist(): {}".format(np.int32(self.spatial_shape)[::-1].tolist()))
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=features,
            indices=data_dict['scale_1']['coors'].int(),
            spatial_shape=np.int32(self.spatial_shape)[::-1].tolist(),
            batch_size=data_dict['batch_size']
        )

        data_dict['coors'] = data_dict['scale_1']['coors']
        data_dict['coors_inv'] = data_dict['scale_1']['coors_inv']
        data_dict['full_coors'] = data_dict['scale_1']['full_coors']
        # stop_here()

        return data_dict