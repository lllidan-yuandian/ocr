import torch
import config as cfg
import torch.nn.functional as F

class RPN(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FasterRCNN, self).__init__()
        self.vgg = VGG()
        self.rpn = RPN()
        self.a =1

    def forward(self, x):
        base_feat = self.vgg(x)
        rpn_result = self.rpn(base_feat,0,0,0)
        return rpn_result


class RPN(torch.nn.Module):
    def __int__(self, din):
        super(RPN, self).__init__()
        self.din = din
        self.rpn_cov = torch.nn.Conv2d(self.din, 512, 3 , 1 ,1, bias=True)

        self.cls_score_size = len(cfg.anchor_scales) * len(cfg.anchor_ratios)*2
        self.cls_score = torch.nn.Conv2d(512, self.cls_score_size, 1, 1, 0)
        self.bbox_size = len(cfg.anchor_scales) * len(cfg.anchor_ratios) * 4
        self.bbox = torch.nn.Conv2d(512, self.bbox_size, 1, 1, 0)


    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(float(input_shape[1]*input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, input):
        batch_size = base_feat.size(0)

        rpn_conv1 = F.relu(self.rpn_cov(base_feat), inplace=True)
        rpn_cls_score = self.cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        rpn_bbox_pred = self.bbox(rpn_conv1)

        cfg_key = 'train' if self.training else 'test'














