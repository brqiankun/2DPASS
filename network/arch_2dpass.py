import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from network.basic_block import Lovasz_loss
from network.baseline import get_model as SPVCNN
from network.base_model import LightningBaseModel
from network.basic_block import ResNetFCN

import logging
logging.basicConfig(format='%(pathname)s->%(lineno)d: %(message)s', level=logging.INFO)
def stop_here():
    raise RuntimeError("🚀" * 5 + "-stop-" + "🚀" * 5)

class xModalKD(nn.Module):
    def __init__(self,config):
        super(xModalKD, self).__init__()
        self.hiden_size = config['model_params']['hiden_size']
        self.scale_list = config['model_params']['scale_list']
        self.num_classes = config['model_params']['num_classes']
        self.lambda_xm = config['train_params']['lambda_xm']
        self.lambda_seg2d = config['train_params']['lambda_seg2d']
        self.num_scales = len(self.scale_list)

        self.multihead_3d_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_3d_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )

        self.multihead_fuse_classifier = nn.ModuleList()
        for i in range(self.num_scales):
            self.multihead_fuse_classifier.append(
                nn.Sequential(
                    nn.Linear(self.hiden_size, 128),
                    nn.ReLU(True),
                    nn.Linear(128, self.num_classes))
            )
        self.leaners = nn.ModuleList()
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()
        for i in range(self.num_scales):
            self.leaners.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))
            self.fcs1.append(nn.Sequential(nn.Linear(self.hiden_size * 2, self.hiden_size)))
            self.fcs2.append(nn.Sequential(nn.Linear(self.hiden_size, self.hiden_size)))

        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(weight=seg_labelweights, ignore_index=config['dataset_params']['ignore_label'])
        self.lovasz_loss = Lovasz_loss(ignore=config['dataset_params']['ignore_label'])

    @staticmethod
    def p2img_mapping(pts_fea, p2img_idx, batch_idx):
        img_feat = []
        for b in range(batch_idx.max()+1):
            logging.info("pts_fea.shape: {}".format(pts_fea.shape))
            logging.info("p2img_idx[{}].shape: {}".format(b, p2img_idx[b].shape))
            logging.info("batch_idx.shape: {}".format(batch_idx.shape))
            img_feat.append(pts_fea[batch_idx == b][p2img_idx[b]])
        return torch.cat(img_feat, 0)    # 一个batch中可以投影到图像的所有点云特征进行拼接

    @staticmethod
    def voxelize_labels(labels, full_coors):
        lbxyz = torch.cat([labels.reshape(-1, 1), full_coors], dim=-1)   # [label, batch_idx, x, y, z]
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)  # 筛选掉相同batch，相同体素内，相同label的点
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]  # 筛选得到共有多少体素被占据，(每个体素内可能有多个label)
        label_ind = torch_scatter.scatter_max(count, inv_ind)[1]    # 对同一个体素内不同类别的label求max得到最多的label种类, label_ind为最大label数量对应的在count中的idx
        logging.info("lbxyz.shape: {}".format(lbxyz.shape))
        logging.info("unq_lbxyz.shape: {}".format(unq_lbxyz.shape))
        logging.info("count.shape: {}".format(count.shape))
        logging.info("inv_ind.shape: {}".format(inv_ind.shape))
        logging.info("label_ind.shape: {}".format(label_ind.shape))
        labels = unq_lbxyz[:, 0][label_ind]    # 得到每个体素的label(按照体素内label所属点数最多的label进行决定)
        logging.info("labels.shape: {}".format(labels.shape))
        # stop_here()
        return labels

    def seg_loss(self, logits, labels):
        ce_loss = self.ce_loss(logits, labels)
        lovasz_loss = self.lovasz_loss(F.softmax(logits, dim=1), labels)
        return ce_loss + lovasz_loss

    def fusion_to_single_KD(self, data_dict, idx):
        batch_idx = data_dict['batch_idx']
        point2img_index = data_dict['point2img_index']
        last_scale = self.scale_list[idx - 1] if idx > 0 else 1
        img_feat = data_dict['img_scale{}'.format(self.scale_list[idx])]
        pts_feat = data_dict['layer_{}'.format(idx)]['pts_feat']
        coors_inv = data_dict['scale_{}'.format(last_scale)]['coors_inv']

        logging.info("img_scale{} img_feat.shape: {}".format(self.scale_list[idx], data_dict['img_scale{}'.format(self.scale_list[idx])].shape))
        logging.info("pts_feat: layer_{} pts_feat.shape: {}".format(idx, data_dict['layer_{}'.format(idx)]['pts_feat'].shape))
        logging.info("scale_{} corrs_inv.shape: {}".format(last_scale, coors_inv.shape))
        # stop_here()
        # 3D prediction
        pts_pred_full = self.multihead_3d_classifier[idx](pts_feat)
        logging.info("pts_pred_full.shape: {}".format(pts_pred_full.shape))

        # correspondence
        pts_label_full = self.voxelize_labels(data_dict['labels'], data_dict['layer_{}'.format(idx)]['full_coors'])  # 得到每个体素的label(根据点云数量最多的label进行确定)
        # 点云特征可以投影到图像的部分,先将点云特征按照体素(体素内的所有点特征相同)恢复到所有点，在按照点到图像像素的对应关系得到可以与图像像素对应的特征
        pts_feat = self.p2img_mapping(pts_feat[coors_inv], point2img_index, batch_idx)
        logging.info("pts_feat.shape: {}".format(pts_feat.shape))
        pts_pred = self.p2img_mapping(pts_pred_full[coors_inv], point2img_index, batch_idx)    # 点云预测结果可以投影到图像的部分
        logging.info("pts_pred.shape: {}".format(pts_pred.shape))

        # modality fusion
        feat_learner = F.relu(self.leaners[idx](pts_feat))
        logging.info("feat_learner.shape: {}".format(feat_learner.shape))
        feat_cat = torch.cat([img_feat, feat_learner], 1)
        logging.info("feat_cat.shape: {}".format(feat_cat.shape))
        feat_cat = self.fcs1[idx](feat_cat)
        feat_weight = torch.sigmoid(self.fcs2[idx](feat_cat))
        fuse_feat = F.relu(feat_cat * feat_weight)
        logging.info("fuse_feat.shape: {}".format(fuse_feat.shape))

        # fusion prediction
        fuse_pred = self.multihead_fuse_classifier[idx](fuse_feat)
        logging.info("fuse_pred.shape: {}".format(fuse_pred.shape))

        # Segmentation Loss
        seg_loss_3d = self.seg_loss(pts_pred_full, pts_label_full)    # 可以投影到图像范围的纯点云特征分类loss
        logging.info("data_dict['img_label'].shape: {}".format(data_dict['img_label'].shape))
        seg_loss_2d = self.seg_loss(fuse_pred, data_dict['img_label'])
        loss = seg_loss_3d + seg_loss_2d * self.lambda_seg2d / self.num_scales

        # KL divergence
        # 可以投影到图像范围内的纯点云预测结果和点云图像融合特征预测结果求loss
        # 蒸馏，使图像范围内的纯点云预测结果接近融合预测结果
        xm_loss = F.kl_div(
            F.log_softmax(pts_pred, dim=1),
            F.softmax(fuse_pred.detach(), dim=1),
        )
        loss += xm_loss * self.lambda_xm / self.num_scales

        # stop_here()

        return loss, fuse_feat

    def forward(self, data_dict):
        loss = 0
        img_seg_feat = []

        for idx in range(self.num_scales):
            singlescale_loss, fuse_feat = self.fusion_to_single_KD(data_dict, idx)
            logging.info("scale_{}: fuse_feat.shape: {}".format(self.scale_list[idx], fuse_feat.shape))
            img_seg_feat.append(fuse_feat)
            loss += singlescale_loss

        logging.info("torch.cat(img_seg_feat, 1).shape: {}".format(torch.cat(img_seg_feat, 1).shape))
        img_seg_logits = self.classifier(torch.cat(img_seg_feat, 1))
        logging.info("img_seg_logits.shape: {}".format(img_seg_logits.shape))
        logging.info("data_dict['img_label'].shape: {}".format(data_dict['img_label'].shape))
        loss += self.seg_loss(img_seg_logits, data_dict['img_label'])
        # stop_here()
        data_dict['loss'] += loss

        # stop_here()

        return data_dict


class get_model(LightningBaseModel):
    def __init__(self, config):
        super(get_model, self).__init__(config)
        self.save_hyperparameters()
        self.baseline_only = config.baseline_only
        self.num_classes = config.model_params.num_classes
        self.hiden_size = config.model_params.hiden_size
        self.lambda_seg2d = config.train_params.lambda_seg2d
        self.lambda_xm = config.train_params.lambda_xm
        self.scale_list = config.model_params.scale_list
        self.num_scales = len(self.scale_list)

        self.model_3d = SPVCNN(config)
        if not self.baseline_only:
            self.model_2d = ResNetFCN(
                backbone=config.model_params.backbone_2d,
                pretrained=config.model_params.pretrained2d,
                config=config
            )
            self.fusion = xModalKD(config)
        else:
            print('Start vanilla training!')

    def forward(self, data_dict):
        # 3D network
        # for k in data_dict.keys():
        #     logging.info(k)
        # stop_here()
        data_dict = self.model_3d(data_dict)   # 提取点云特征并进行点云分割loss计算
        # for k in data_dict.keys():
        #     logging.info(k)
        # stop_here()

        # training with 2D network
        if self.training and not self.baseline_only:  # 设置为True，进行在测试模式下测试2d network
            data_dict = self.model_2d(data_dict)   # 提取出图像特征
            # for k in data_dict.keys():
            #     logging.info(k)
            stop_here()
            data_dict = self.fusion(data_dict)   # 对能够通过投影对应的图像和点云特征进行融合后预测得到loss
            # for k in data_dict.keys():
            #     logging.info(k)
            # stop_here()

        return data_dict