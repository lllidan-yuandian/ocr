import torch
import config as cfg

class RPN(torch.nn.Module):
    def __int__(self, din):
        super(RPN, self).__init__()
        self.din = din
        self.rpn_cov = torch.nn.Conv2d(self.din, 512, 3 , 1 ,1, bias=True)

        self.cls_score_size = len(cfg.anchor_scales) * len(cfg.anchor_ratios)*2
        self.cls_score = torch.nn.Conv2d(512, self.cls_score_size, 1, 1, 0)
        self.bbox_size = len(cfg.anchor_scales) * len(cfg.anchor_ratios) * 4
        self.bbox = torch.nn.Conv2d(512, self.bbox_size, 1, 1, 0)

    def forward(self, input):









