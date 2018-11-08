# -*- coding: utf-8 -*-
import torch
from generate_anchors import *
from config import *
import numpy as np


class ProposalLayer(torch.nn.Module):
    def __init__(self, feat_stride, scales, ratios):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ProposalLayer, self).__init__()

        self.feat_stride = feat_stride
        self.anchors = torch.from_numpy(
            generate_anchors(scales=scales, ratios=ratios)
        ).float()
        self.num_anchors = self.anchors.size(0)


    def forward(self, input):
        scores = input[0][:, self.num_anchors:, :, :]
        bbox_deltas = input[1]
        im_info = input[2]
        cfg_key = input[3]
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self.feat_stride
        shift_y = np.arange(0, feat_height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        shifts = torch.from_numpy(shifts)
        shifts = shifts.contiguous().type_as(scores).float()

        A =  self.num_anchors
        K = shifts.size(0)

        self.anchors = self.anchors.type_as(scores)

        anchors = (self.anchors.view(1,A,4) + shifts.view(K,1,4)).numpy()
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K*A, 4)


        return input