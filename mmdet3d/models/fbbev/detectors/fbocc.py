# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.runner import force_fp32
import os
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors import CenterPoint
from mmdet3d.models.builder import build_head, build_neck
import numpy as np
import copy 
import spconv.pytorch as spconv
from tqdm import tqdm 
from mmdet3d.models.fbbev.utils import run_time
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from mmcv.runner import get_dist_info
from mmdet.core import reduce_mean
import mmcv
from mmdet3d.datasets.utils import nuscenes_get_rt_matrix
from mmdet3d.core.bbox import box_np_ops # , corner_to_surfaces_3d, points_in_convex_polygon_3d_jit



def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res


@DETECTORS.register_module()
class FBOCC(CenterPoint):

    def __init__(self, 
                 # BEVDet components
                 forward_projection=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 # BEVFormer components
                 backward_projection=None,
                 # FB-BEV components
                 frpn=None,
                 # depth_net
                 depth_net=None,
                 # HPNet
                 hardness_net=None,
                 # Instance_fusion
                 instance_fusion=None,
                 #scene_hpnet
                 scene_hpnet=None,
                 # occupancy head
                 occupancy_head=None,
                 # other settings.
                 use_depth_supervision=False,
                 readd=False,
                 fix_void=False,
                 occupancy_save_path=None,
                 do_history=True,
                 interpolation_mode='bilinear',
                 history_cat_num=16,
                 history_cat_conv_out_channels=None,
                 single_bev_num_channels=80,

                  **kwargs):
        super(FBOCC, self).__init__(**kwargs)
        self.fix_void = fix_void
      
        # BEVDet init
        self.forward_projection = builder.build_neck(forward_projection) if forward_projection else None
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone) if img_bev_encoder_backbone else None
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck) if img_bev_encoder_neck else None

        # BEVFormer init
        self.backward_projection = builder.build_head(backward_projection) if backward_projection else None
    
        # FB-BEV init
        if not self.forward_projection: assert not frpn, 'frpn relies on LSS'
        self.frpn = builder.build_head(frpn) if frpn else None

        # Depth Net
        self.depth_net = builder.build_head(depth_net) if depth_net else None

        # hardness_net
        self.hardness_net = builder.build_head(hardness_net) if hardness_net else None

        # instance_fusion
        self.instance_fusion = builder.build_head(instance_fusion) if instance_fusion else None
        self.history_instances = None  # 存储历史实例
        self.is_first_frame = True  # 第一帧标志

        #scene_hpnet
        self.scene_hpnet = builder.build_head(scene_hpnet) if scene_hpnet else None

        # Occupancy Head
        self.occupancy_head = builder.build_head(occupancy_head) if occupancy_head else None

        
        self.readd = readd # fuse voxel features and bev features
        
        self.use_depth_supervision = use_depth_supervision
        
        self.occupancy_save_path = occupancy_save_path # for saving data\for submitting to test server

        # Deal with history
        self.single_bev_num_channels = single_bev_num_channels
        self.do_history = do_history
        self.interpolation_mode = interpolation_mode
        self.history_cat_num = history_cat_num
        self.history_cam_sweep_freq = 0.5 # seconds between each frame
        history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                         if history_cat_conv_out_channels is not None 
                                         else self.single_bev_num_channels)
        ## Embed each sample with its relative temporal offset with current timestep
        conv = nn.Conv2d if self.forward_projection.nx[-1] == 1 else nn.Conv3d
        self.history_keyframe_time_conv = nn.Sequential(
             conv(self.single_bev_num_channels + 1,
                     self.single_bev_num_channels,
                     kernel_size=1,
                     padding=0,
                     stride=1),
             nn.SyncBatchNorm(self.single_bev_num_channels),
             nn.ReLU(inplace=True))
        ## Then concatenate and send them through an MLP.
        self.history_keyframe_cat_conv = nn.Sequential(
            conv(self.single_bev_num_channels * (self.history_cat_num + 1),
                    history_cat_conv_out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1),
            nn.SyncBatchNorm(history_cat_conv_out_channels),
            nn.ReLU(inplace=True))
        self.history_sweep_time = None
        self.history_bev = None
        self.history_bev_before_encoder = None
        self.history_seq_ids = None
        self.history_forward_augs = None
        self.count = 0

    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
      
        x = self.img_backbone(imgs)
       
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
      
        return x

    @force_fp32()
    def bev_encoder(self, x):
        if self.with_specific_component('img_bev_encoder_backbone'):
            x = self.img_bev_encoder_backbone(x)
        
        if self.with_specific_component('img_bev_encoder_neck'):
            x = self.img_bev_encoder_neck(x)
        
        if type(x) not in [list, tuple]:
             x = [x]

        return x
        
    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda): # align features with 3d shift

        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)

        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)

        ## Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)


        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum()>0:
            if voxel_feat:    
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1)
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]


        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack(
            (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.forward_projection.dx[0]
        feat2bev[1, 1] = self.forward_projection.dx[1]
        feat2bev[2, 2] = self.forward_projection.dx[2]
        feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
        feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
        feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
        # feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)

        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations.
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                   @ torch.inverse(forward_augs) @ feat2bev)

        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

        # normalize and sample
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:,:,:,:, :3,0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        

        tmp_bev = self.history_bev
        if voxel_feat: 
            n, mc, z, h, w = tmp_bev.shape
            tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
        sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode=self.interpolation_mode)

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)

        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 

        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        if voxel_feat:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None, None].repeat(
                1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x Z x H x W
        else:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x H x W

        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        
        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        return feats_to_return.clone()


    def extract_img_bev_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""

        return_map = {}
        mlp_input = None
        context = self.image_encoder(img[0])
        cam_params = img[1:7]
        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            context, depth = self.depth_net(context, mlp_input)
            return_map['depth'] = depth
            return_map['context'] = context
        else:
            context=None
            depth=None

        if self.with_specific_component('hardness_net'):
            hardness = self.hardness_net(depth.detach(), context.detach(), mlp_input.detach())
            return_map['cam_hardness'] = hardness
        kwargs['hardness'] = hardness
        
        if self.with_specific_component('forward_projection'):
            bev_feat, bev_hardness = self.forward_projection(cam_params, context, depth, **kwargs)
            return_map['cam_params'] = cam_params
            return_map['bev_hardness'] = bev_hardness
        else:
            bev_feat = None
        
        if self.with_specific_component('frpn'): # not used in FB-OCC
            bev_mask_logit = self.frpn(bev_feat)
            bev_mask = bev_mask_logit.sigmoid() > self.frpn.mask_thre
            
            if bev_mask.requires_grad: # during training phase
                gt_bev_mask = kwargs['gt_bev_mask'].to(torch.bool)
                bev_mask = gt_bev_mask | bev_mask
            return_map['bev_mask_logit'] = bev_mask_logit    
        else:
            bev_mask = None

        if self.with_specific_component('backward_projection'):

            bev_feat_refined = self.backward_projection([context],
                                        img_metas,
                                        lss_bev=bev_feat.mean(-1),
                                        cam_params=cam_params,
                                        bev_mask=bev_mask,
                                        gt_bboxes_3d=None, # debug
                                        pred_img_depth=depth)  
                                        
            if self.readd:
                bev_feat = bev_feat_refined[..., None] + bev_feat
            else:
                bev_feat = bev_feat_refined

        # Fuse History
        bev_feat = self.fuse_history(bev_feat, img_metas, img[6])

        if self.scene_hpnet is not None:
            scene_hardness = self.scene_hpnet(bev_feat.detach())
            return_map['scene_hardness'] = scene_hardness

        if self.instance_fusion is not None:
            bev_feat_inst_refined = self.process_instance_fusion(
                bev_feat, context, bev_hardness.detach(), img_metas, return_map, gt_occupancy = kwargs['gt_occupancy'])

        with torch.no_grad():
            bev_feat = self.bev_encoder(bev_feat)
            return_map['img_bev_feat'] = bev_feat

        bev_feat_inst_refined = self.bev_encoder(bev_feat_inst_refined)
        return_map['img_bev_feat_inst_refined'] = bev_feat_inst_refined

        return return_map

    def extract_lidar_bev_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        bev_feat = self.pts_middle_encoder(voxel_features, coors, batch_size)
        bev_feat = self.bev_encoder(bev_feat)
        return bev_feat

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        results={}
        if img is not None and self.with_specific_component('image_encoder'):
            results.update(self.extract_img_bev_feat(img, img_metas, **kwargs))
        if points is not None and self.with_specific_component('pts_voxel_encoder'):
            results['lidar_bev_feat'] = self.extract_lidar_bev_feat(points, img, img_metas)

        return results


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_occupancy_flow=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """


        results= self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()

        if  self.with_pts_bbox:
            losses_pts = self.forward_pts_train(results['img_bev_feat'], gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
            losses.update(losses_pts)
            
        if self.with_specific_component('occupancy_head'):
            losses_occupancy = self.occupancy_head.forward_train(results['img_bev_feat'], results=results, gt_occupancy=kwargs['gt_occupancy'], gt_occupancy_flow=gt_occupancy_flow, mark=True)
            losses.update(losses_occupancy)
            
            losses_occupancy_refined = self.occupancy_head.forward_train(results['img_bev_feat_inst_refined'], results=results,
                                                                 gt_occupancy=kwargs['gt_occupancy'],
                                                                 gt_occupancy_flow=gt_occupancy_flow, mark=False)
            losses.update(losses_occupancy_refined)

        if self.with_specific_component('frpn'):
            losses_mask = self.frpn.get_bev_mask_loss(kwargs['gt_bev_mask'], results['bev_mask_logit'])
            losses.update(losses_mask)

        if self.use_depth_supervision and self.with_specific_component('depth_net'):
            loss_depth = self.depth_net.get_depth_loss(kwargs['gt_depth'], results['depth'])
            losses.update(loss_depth)

        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        self.do_history = True
        if img_inputs is not None:
            for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
                if not isinstance(var, list) :
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))        
            num_augs = len(img_inputs)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(img_inputs), len(img_metas)))

            if num_augs==1 and not img_metas[0][0].get('tta_config', dict(dist_tta=False))['dist_tta']:
                return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
            else:
                return self.aug_test(points, img_metas, img_inputs, **kwargs)
        
        elif points is not None:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        
    def aug_test(self,points,
                    img_metas,
                    img_inputs=None,
                    visible_mask=[None],
                    **kwargs):
        """Test function without augmentaiton."""
        assert False
        return None

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    visible_mask=[None],
                    return_raw_occ=False,
                    **kwargs):
        """Test function without augmentaiton."""
        results = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        

        bbox_list = [dict() for _ in range(len(img_metas))]
        
        if  self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(results['img_bev_feat'], img_metas, rescale=rescale)
        else:
            bbox_pts = [None for _ in range(len(img_metas))]


        if self.with_specific_component('occupancy_head'):
            pred_occupancy = self.occupancy_head(results['img_bev_feat'], results=results, **kwargs)['output_voxels'][0]
            #pred_occupancy = self.occupancy_head(results['img_bev_feat_inst_refined'], results=results, **kwargs)['output_voxels'][0]


            pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
            if self.fix_void:
                pred_occupancy = pred_occupancy[..., 1:]     
            pred_occupancy = pred_occupancy.softmax(-1)


            # convert to CVPR2023 Format
            pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
            pred_occupancy = torch.flip(pred_occupancy, [2])
            pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
            pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)
            
            if return_raw_occ:
                pred_occupancy_category = pred_occupancy
            else:
                pred_occupancy_category = pred_occupancy.argmax(-1) 
            

            # # do not change the order
            # if self.occupancy_save_path is not None:
            #     scene_name = img_metas[0]['scene_name']
            #     sample_token = img_metas[0]['sample_idx']
            #     mask_camera = visible_mask[0][0]
            #     masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
            #     save_path = os.path.join(self.occupancy_save_path, 'occupancy_pred', scene_name+'_'+sample_token)
            #     np.savez_compressed(save_path, pred=masked_pred_occupancy, sample_token=sample_token) 


            # For test server
            if self.occupancy_save_path is not None:
                    scene_name = img_metas[0]['scene_name']
                    sample_token = img_metas[0]['sample_idx']
                    # mask_camera = visible_mask[0][0]
                    # masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
                    save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
                    save_path = os.path.join(self.occupancy_save_path, 'occupancy_pred', f'{sample_token}.npz')
                    np.savez_compressed(save_path, save_pred_occupancy.astype(np.uint8)) 

            pred_occupancy_category= pred_occupancy_category.cpu().numpy()

        else:
            pred_occupancy_category =  None

        if results.get('bev_mask_logit', None) is not None:
            pred_bev_mask = results['bev_mask_logit'].sigmoid() > 0.5
            iou = IOU(pred_bev_mask.reshape(1, -1), kwargs['gt_bev_mask'][0].reshape(1, -1)).cpu().numpy()
        else:
            iou = None

        assert len(img_metas) == 1
        for i, result_dict in enumerate(bbox_list):
            result_dict['pts_bbox'] = bbox_pts[i]
            result_dict['iou'] = iou
            result_dict['pred_occupancy'] = pred_occupancy_category
            result_dict['index'] = img_metas[0]['index']
        return bbox_list


    # def simple_test(self,
    #                 points,
    #                 img_metas,
    #                 img=None,
    #                 rescale=False,
    #                 visible_mask=[None],
    #                 return_raw_occ=False,
    #                 **kwargs):
    #     """Test function without augmentaiton."""
    #     results = self.extract_feat(
    #         points, img=img, img_metas=img_metas, **kwargs)

    #     bbox_list = [dict() for _ in range(len(img_metas))]

    #     if self.with_pts_bbox:
    #         bbox_pts = self.simple_test_pts(results['img_bev_feat'], img_metas, rescale=rescale)
    #     else:
    #         bbox_pts = [None for _ in range(len(img_metas))]

    #     if self.with_specific_component('occupancy_head'):
    #         # 获取包含所有信号的occ结果
    #         occ_res = self.occupancy_head(
    #             results['img_bev_feat_inst_refined'],
    #             results=results,
    #             hardness_pred=results.get('bev_hardness'),
    #             scene_hardness_pred=results.get('scene_hardness'),
    #             visible_mask=visible_mask[0] if visible_mask[0] is not None else None,
    #             **kwargs
    #         )

    #         pred_occupancy = occ_res['output_voxels'][0]

    #         # 原始occupancy处理逻辑
    #         pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]  # [H, W, D, C]
    #         if self.fix_void:
    #             pred_occupancy = pred_occupancy[..., 1:]
    #         pred_occupancy = pred_occupancy.softmax(-1)  # [H, W, D, C]

    #         # convert to CVPR2023 Format
    #         pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)  # [C, D, H, W]
    #         pred_occupancy = torch.flip(pred_occupancy, [2])  # 上下翻转 [C, D, H, W]
    #         pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])  # 旋转-90度 [C, D, H, W]
    #         pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)  # [H, W, D, C]

    #         if return_raw_occ:
    #             pred_occupancy_category = pred_occupancy
    #         else:
    #             pred_occupancy_category = pred_occupancy.argmax(-1)

    #         pred_occupancy_category = pred_occupancy_category.cpu().numpy()

    #         # ==================== 定义转换函数 ====================
    #         def convert_tensor_to_cvpr_format(tensor, signal_name):
    #             """
    #             将张量转换为CVPR2023格式
    #             注意：occupancy的转换流程是 [H, W, D, C] → [C, D, H, W] → flip → rot90 → [H, W, D, C]
    #             """
    #             if tensor is None:
    #                 return None

    #             # 确保是numpy格式
    #             if torch.is_tensor(tensor):
    #                 tensor_np = tensor.detach().cpu().numpy()
    #             else:
    #                 tensor_np = tensor

    #             # 获取张量形状和维度
    #             original_shape = tensor_np.shape
    #             ndim = tensor_np.ndim

    #             # 根据信号类型和形状确定转换方式
    #             if ndim == 3:  # [H, W, D] 格式
    #                 # 应用与occupancy相同的空间变换
    #                 tensor_np = tensor_np.transpose(2, 0, 1)  # [D, H, W]
    #                 tensor_np = np.flip(tensor_np, axis=1)  # 上下翻转
    #                 tensor_np = np.rot90(tensor_np, -1, axes=(1, 2))  # 旋转-90度
    #                 tensor_np = tensor_np.transpose(1, 2, 0)  # [H, W, D]

    #             elif ndim == 4:  # 需要判断是 [B, H, W, D], [H, W, D, C], 还是 [C, H, W, D]
    #                 if original_shape[0] == 1:  # [1, H, W, D] - 通常是batch维度
    #                     # 去掉batch维度，变成 [H, W, D]，然后按照3D处理
    #                     tensor_np = tensor_np[0]
    #                     tensor_np = tensor_np.transpose(2, 0, 1)  # [D, H, W]
    #                     tensor_np = np.flip(tensor_np, axis=1)  # 上下翻转
    #                     tensor_np = np.rot90(tensor_np, -1, axes=(1, 2))  # 旋转-90度
    #                     tensor_np = tensor_np.transpose(1, 2, 0)  # [H, W, D]

    #                 elif original_shape[-1] <= 20:  # 假设类别数小于等于20，则是 [H, W, D, C]
    #                     # 按照occupancy的转换方式：HWDC → CDHW → flip → rot90 → HWDC
    #                     tensor_np = tensor_np.transpose(3, 2, 0, 1)  # [C, D, H, W]
    #                     tensor_np = np.flip(tensor_np, axis=2)  # 上下翻转
    #                     tensor_np = np.rot90(tensor_np, -1, axes=(2, 3))  # 旋转-90度
    #                     tensor_np = tensor_np.transpose(2, 3, 1, 0)  # [H, W, D, C]

    #                 else:  # 可能是 [C, H, W, D] 或 [B, C, H, W, D] 去掉batch后是 [C, H, W, D]
    #                     # 按照CHWD格式转换
    #                     tensor_np = tensor_np.transpose(0, 3, 1, 2)  # [C, D, H, W]
    #                     tensor_np = np.flip(tensor_np, axis=2)  # 上下翻转
    #                     tensor_np = np.rot90(tensor_np, -1, axes=(2, 3))  # 旋转-90度
    #                     tensor_np = tensor_np.transpose(2, 3, 1, 0)  # [H, W, D, C]

    #             else:
    #                 print(f"Warning: Unsupported tensor dimension for {signal_name}: {ndim}, shape: {original_shape}")
    #                 # 尝试降维处理
    #                 if ndim == 5:  # [B, C, H, W, D]
    #                     # 取第一个batch，然后按照4D处理
    #                     tensor_np = tensor_np[0]
    #                     if tensor_np.shape[0] <= 20:  # 如果是类别维度
    #                         tensor_np = tensor_np.transpose(0, 3, 1, 2)  # [C, D, H, W]
    #                     else:
    #                         tensor_np = tensor_np.transpose(1, 3, 2, 0)  # 尝试重新排列

    #                     tensor_np = np.flip(tensor_np, axis=2)  # 上下翻转
    #                     tensor_np = np.rot90(tensor_np, -1, axes=(2, 3))  # 旋转-90度
    #                     tensor_np = tensor_np.transpose(2, 3, 1, 0)  # [H, W, D, C]

    #             return tensor_np

    #         # ==================== 处理并保存所有信号 ====================
    #         if self.occupancy_save_path is not None:
    #             scene_name = img_metas[0]['scene_name']
    #             sample_token = img_metas[0]['sample_idx']

    #             # 1. 保存occupancy预测（原始逻辑）
    #             save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
    #             occ_save_path = os.path.join(self.occupancy_save_path, 'occupancy_pred', f'{sample_token}.npz')
    #             np.savez_compressed(occ_save_path, save_pred_occupancy.astype(np.uint8))

    #             # 2. 创建监控信号保存目录
    #             monitor_save_dir = os.path.join(self.occupancy_save_path, 'monitor_signals')
    #             os.makedirs(monitor_save_dir, exist_ok=True)

    #             # ==================== 新增：创建visible信号保存目录 ====================
    #             visible_save_dir = os.path.join(self.occupancy_save_path, 'visible_signals')
    #             os.makedirs(visible_save_dir, exist_ok=True)

    #             # 3. 处理并保存各个信号
    #             monitor_data = {}
    #             visible_monitor_data = {}

    #             # 处理体素级困难度
    #             voxel_hardness = occ_res.get('hardness_pred')
    #             if voxel_hardness is not None:
    #                 voxel_hardness_cvpr = convert_tensor_to_cvpr_format(voxel_hardness, 'voxel_hardness')
    #                 monitor_data['voxel_hardness'] = voxel_hardness_cvpr.astype(np.float32)

    #             # 处理场景级困难度
    #             scene_hardness = occ_res.get('scene_hardness_pred')
    #             if scene_hardness is not None:
    #                 scene_hardness_cvpr = convert_tensor_to_cvpr_format(scene_hardness, 'scene_hardness')
    #                 monitor_data['scene_hardness'] = scene_hardness_cvpr.astype(np.float32)

    #             # 处理全局困难度
    #             global_hardness = occ_res.get('global_hardness')
    #             if global_hardness is not None:
    #                 global_hardness_cvpr = convert_tensor_to_cvpr_format(global_hardness, 'global_hardness')
    #                 monitor_data['global_hardness'] = global_hardness_cvpr.astype(np.float32)

    #             # 处理损失分布
    #             loss_dist = occ_res.get('loss_distribution')
    #             if loss_dist is not None:
    #                 loss_dist_cvpr = convert_tensor_to_cvpr_format(loss_dist, 'loss_distribution')
    #                 monitor_data['loss_distribution'] = loss_dist_cvpr.astype(np.float32)

    #             # 保存元数据
    #             monitor_data['sample_token'] = sample_token
    #             monitor_data['scene_name'] = scene_name
    #             monitor_data['grid_shape'] = pred_occupancy_category.shape

    #             # 保存监控信号文件
    #             monitor_save_path = os.path.join(monitor_save_dir, f'{sample_token}_signals.npz')
    #             np.savez_compressed(monitor_save_path, **monitor_data)

    #             # ==================== 新增：保存visible监控信号 ====================
    #             # 获取visible_monitor_stats
    #             visible_monitor_stats = occ_res.get('visible_monitor_stats', {})
    #             if visible_monitor_stats:
    #                 # 转换numpy类型
    #                 def convert_to_serializable(obj):
    #                     if isinstance(obj, torch.Tensor):
    #                         if obj.numel() == 1:
    #                             return obj.cpu().item()
    #                         else:
    #                             return obj.cpu().numpy()
    #                     elif isinstance(obj, (np.ndarray, np.generic)):
    #                         return obj.tolist() if hasattr(obj, 'tolist') else obj
    #                     else:
    #                         return obj

    #                 for key, value in visible_monitor_stats.items():
    #                     visible_monitor_data[key] = convert_to_serializable(value)
                    
    #                 # 添加元数据
    #                 visible_monitor_data['sample_token'] = sample_token
    #                 visible_monitor_data['scene_name'] = scene_name
                    
    #                 # 保存visible监控信号文件
    #                 visible_save_path = os.path.join(visible_save_dir, f'{sample_token}_visible_signals.npz')
    #                 np.savez_compressed(visible_save_path, **visible_monitor_data)
                    
    #                 # ==================== 新增：保存visible监控统计为JSON ====================
    #                 visible_stats_dir = os.path.join(self.occupancy_save_path, 'visible_stats')
    #                 os.makedirs(visible_stats_dir, exist_ok=True)
                    
    #                 # 将统计信息保存为JSON
    #                 import json
    #                 visible_stats_path = os.path.join(visible_stats_dir, f'{sample_token}_visible_stats.json')
    #                 with open(visible_stats_path, 'w') as f:
    #                     json.dump(visible_monitor_data, f, indent=2, ensure_ascii=False)
                    
    #                 print(f"Saved visible monitor signals to: {visible_save_path}")
    #                 print(f"  Contains: {list(visible_monitor_stats.keys())}")

    #             # ==================== 新增：保存cam_hardness ====================
    #             # 4. 创建cam_hardness保存目录
    #             cam_hardness_dir = os.path.join(self.occupancy_save_path, 'cam_hardness')
    #             os.makedirs(cam_hardness_dir, exist_ok=True)

    #             # 获取cam_hardness
    #             cam_hardness = results.get('cam_hardness')
    #             if cam_hardness is not None:
    #                 # cam_hardness维度: [B, N, C, H, W] = [1, 6, 80, 16, 44]
    #                 if torch.is_tensor(cam_hardness):
    #                     cam_hardness_np = cam_hardness.detach().cpu().numpy()
    #                 else:
    #                     cam_hardness_np = cam_hardness

    #                 # 去掉batch维度，保存为 [N, C, H, W] = [6, 80, 16, 44]
    #                 if cam_hardness_np.shape[0] == 1:
    #                     cam_hardness_np = cam_hardness_np[0]

    #                 # 保存为.npz文件（与其他信号格式一致）
    #                 cam_hardness_npz_path = os.path.join(cam_hardness_dir, f'{sample_token}_cam_hardness.npz')
    #                 np.savez_compressed(cam_hardness_npz_path, cam_hardness=cam_hardness_np.astype(np.float32))

    #                 # 保存cam_hardness的统计信息
    #                 cam_hardness_stats = {
    #                     'sample_token': sample_token,
    #                     'scene_name': scene_name,
    #                     'shape': cam_hardness_np.shape,
    #                     'mean_per_camera': cam_hardness_np.mean(axis=(1, 2, 3)).tolist(),
    #                     'std_per_camera': cam_hardness_np.std(axis=(1, 2, 3)).tolist(),
    #                     'min_per_camera': cam_hardness_np.min(axis=(1, 2, 3)).tolist(),
    #                     'max_per_camera': cam_hardness_np.max(axis=(1, 2, 3)).tolist(),
    #                 }

    #                 # 保存统计信息到JSON
    #                 import json
    #                 cam_stats_path = os.path.join(cam_hardness_dir, f'{sample_token}_cam_stats.json')
    #                 with open(cam_stats_path, 'w') as f:
    #                     json.dump(cam_hardness_stats, f, indent=2, ensure_ascii=False)

    #             # 5. 保存统计信号为JSON文件
    #             monitor_stats = occ_res.get('monitor_stats', {})
    #             if monitor_stats:
    #                 import json
    #                 stats_save_dir = os.path.join(self.occupancy_save_path, 'monitor_stats')
    #                 os.makedirs(stats_save_dir, exist_ok=True)

    #                 # 添加元数据到统计
    #                 monitor_stats['sample_token'] = sample_token
    #                 monitor_stats['scene_name'] = scene_name
    #                 monitor_stats['grid_shape'] = pred_occupancy_category.shape
    #                 monitor_stats['timestamp'] = img_metas[0].get('timestamp', 'unknown')

    #                 # 转换numpy类型为Python原生类型
    #                 def convert_to_serializable(obj):
    #                     if isinstance(obj, np.ndarray):
    #                         return obj.tolist()
    #                     elif isinstance(obj, (np.integer, np.floating)):
    #                         return obj.item()
    #                     elif isinstance(obj, torch.Tensor):
    #                         return obj.cpu().item() if obj.numel() == 1 else obj.cpu().tolist()
    #                     else:
    #                         return obj

    #                 serializable_stats = {
    #                     k: convert_to_serializable(v) for k, v in monitor_stats.items()
    #                 }

    #                 stats_save_path = os.path.join(stats_save_dir, f'{sample_token}_stats.json')
    #                 with open(stats_save_path, 'w') as f:
    #                     json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

    #                 # 也保存为npz格式
    #                 npz_stats_path = os.path.join(stats_save_dir, f'{sample_token}_stats.npz')
    #                 np.savez_compressed(npz_stats_path, **serializable_stats)

    #                 print(f"Saved monitor stats to: {stats_save_path}")

    #             print(f"Saved monitor signals to: {monitor_save_path}")
    #             print(f"  Contains: {list(monitor_data.keys())}")

    #     if results.get('bev_mask_logit', None) is not None:
    #         pred_bev_mask = results['bev_mask_logit'].sigmoid() > 0.5
    #         iou = IOU(pred_bev_mask.reshape(1, -1), kwargs['gt_bev_mask'][0].reshape(1, -1)).cpu().numpy()
    #     else:
    #         iou = None

    #     assert len(img_metas) == 1
    #     for i, result_dict in enumerate(bbox_list):
    #         result_dict['pts_bbox'] = bbox_pts[i]
    #         result_dict['iou'] = iou
    #         result_dict['pred_occupancy'] = pred_occupancy_category if 'pred_occupancy_category' in locals() else None
    #         result_dict['index'] = img_metas[0]['index']

    #     return bbox_list




















    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        results = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(results['img_bev_feat'])
        return outs

    @force_fp32()
    def process_instance_fusion(self, bev_feat, context, bev_hardness, img_metas, return_map, gt_occupancy):
        bs = bev_feat.shape[0]
        B, C, H, W, D = bev_feat.shape

        if bev_hardness is not None:
            hardness_flat = bev_hardness.reshape(B, -1, 1)
            bev_feat_flat = bev_feat.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

            global_topk_indices = torch.topk(
                hardness_flat.squeeze(-1),
                k=min(2000, hardness_flat.shape[1]),
                dim=1
            )[1]

            visible_topk_indices = None
            if gt_occupancy is not None:
                if isinstance(gt_occupancy, list):
                    gt_occupancy = gt_occupancy[0]
                gt_occ = gt_occupancy.detach()
                visible_mask = (gt_occ != 255).float()
                visible_mask_down = visible_mask.permute(0, 3, 1, 2).float()
                visible_mask_down = F.avg_pool2d(visible_mask_down, kernel_size=2, stride=2)
                
                if visible_mask_down.shape[1] == 16:
                    visible_mask_down = visible_mask_down.reshape(B, 8, 2, 100, 100).mean(dim=2)
                
                visible_mask_down = (visible_mask_down > 0.1).float()
                visible_mask_down = visible_mask_down.permute(0, 2, 3, 1)
                visible_flat = visible_mask_down.reshape(B, -1, 1)
                
                masked_hardness = hardness_flat.clone()
                masked_hardness[visible_flat.squeeze(-1) < 0.5] = 0
                
                visible_topk_indices = torch.topk(
                    masked_hardness.squeeze(-1),
                    k=min(500, hardness_flat.shape[1]),
                    dim=1
                )[1]

            if visible_topk_indices is not None:
                combined_indices_list = []
                for i in range(B):
                    global_set = set(global_topk_indices[i].cpu().numpy())
                    visible_set = set(visible_topk_indices[i].cpu().numpy())
                    union_set = global_set.union(visible_set)
                    union_tensor = torch.tensor(list(union_set), device=bev_feat.device, dtype=torch.long)
                    combined_indices_list.append(union_tensor)
                
                max_len = max([t.shape[0] for t in combined_indices_list])
                topk_indices = torch.zeros(B, max_len, device=bev_feat.device, dtype=torch.long)
                
                for i, idx in enumerate(combined_indices_list):
                    topk_indices[i, :idx.shape[0]] = idx
                
                print(f"Union indices count: {topk_indices.shape[1]}")
            else:
                topk_indices = global_topk_indices

            initial_hard_voxel_feat = torch.gather(
                bev_feat_flat,
                dim=1,
                index=topk_indices.unsqueeze(-1).expand(-1, -1, C)
            )
        else:
            initial_hard_voxel_feat = None
            bev_feat_flat = None
            topk_indices = None

        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence']
            for single_img_metas in img_metas]).to(bev_feat.device)

        if self.history_instances is None:
            self.history_instances = self.instance_fusion.history_inst_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        if start_of_sequence.sum() > 0:
            reset_instances = self.instance_fusion.history_inst_embed.weight.unsqueeze(0).repeat(
                start_of_sequence.sum(), 1, 1)
            self.history_instances[start_of_sequence] = reset_instances

        history_inst_queries = self.history_instances.clone()

        fused_global_inst, updated_history_inst, final_enhanced_voxel, camera_instances = self.instance_fusion(
            context=context,
            hard_voxel_feat=initial_hard_voxel_feat,
            history_inst_queries=history_inst_queries
        )

        self.history_instances = updated_history_inst.detach()

        if final_enhanced_voxel is not None and initial_hard_voxel_feat is not None:
            bev_feat_flat_combined = bev_feat_flat.scatter(
                dim=1,
                index=topk_indices.unsqueeze(-1).expand(-1, -1, C),
                src=final_enhanced_voxel
            )

            bev_feat_reshaped = bev_feat_flat_combined.reshape(B, H, W, D, C)
            bev_feat_refined = bev_feat_reshaped.permute(0, 4, 1, 2, 3)
        else:
            bev_feat_refined = bev_feat

        return_map['fused_global_instances'] = fused_global_inst
        return_map['camera_instances'] = camera_instances
        return_map['updated_history_instances'] = updated_history_inst
        return_map['hard_voxel_indices'] = topk_indices if bev_hardness is not None else None

        return bev_feat_refined