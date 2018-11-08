# -*- coding: utf-8 -*-
import torch
from vgg import *
from rpn import *

class FasterRCNN(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FasterRCNN, self).__init__()
        self.vgg = VGG()
        self.rpn = RPN(512)

        self.a =1

    def forward(self, x):
        base_feat = self.vgg(x)
        rpn_result = self.rpn(base_feat,0,0,0)
        return rpn_result





if __name__ == '__main__':

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(1, 3, 600, 1024)
    model = FasterRCNN()
    # print(model.rpn.a)

    print(model)
    y_pred = model(x)
    print(y_pred)
