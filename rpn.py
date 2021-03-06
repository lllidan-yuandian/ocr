# -*- coding: utf-8 -*-
import torch
from config import *
import torch.nn.functional as F
from proposal_layer import *


class RPN(torch.nn.Module):
    def __init__(self, din):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(RPN, self).__init__()
        self.din = din
        self.feat_stride = cfg.FEAT_STRIDE[0]
        self.anchor_scales = np.array(cfg.anchor_scales)
        self.anchor_ratios = np.array(cfg.anchor_ratios)
        self.rpn_cov = torch.nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)
        self.cls_score_size = len(cfg.anchor_scales) * len(cfg.anchor_ratios) * 2
        self.cls_score = torch.nn.Conv2d(512, self.cls_score_size, 1, 1, 0)
        self.bbox_size = len(cfg.anchor_scales) * len(cfg.anchor_ratios) * 4
        self.bbox = torch.nn.Conv2d(512, self.bbox_size, 1, 1, 0)
        self.rpn_proposal = ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, input):
        batch_size = base_feat.size(0)

        rpn_conv1 = F.relu(self.rpn_cov(base_feat), inplace=True)
        rpn_cls_score = self.cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.cls_score_size)

        rpn_bbox_pred = self.bbox(rpn_conv1)

        cfg_key = 'train' if self.training else 'test'
        rois = self.rpn_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                  im_info, cfg_key))

        return input