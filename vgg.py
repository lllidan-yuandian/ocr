# -*- coding: utf-8 -*-
import torch


class VGG(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(VGG, self).__init__()
        ####################Conv1##############

        self.conv1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.maxpool_1= torch.nn.MaxPool2d(kernel_size=2, stride=2)

        ###################Conv2################
        self.conv2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.conv2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True))
        self.maxpool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


        #################Conv3##################
        self.conv3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        self.conv3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True))
        self.conv3_3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        self.maxpool_3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


        #################Conv4#################
        self.conv4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True)
        )
        self.conv4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True))

        self.conv4_3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True))
        self.maxpool_4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


        ################Conv5#################
        self.conv5_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True))

        self.conv5_2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True))

        self.conv5_3 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True))
        self.maxpool_5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.a =1

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ####################Conv1##############
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool_1(x)


        ###################Conv2################
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.maxpool_2(x)


        #################Conv3##################
        x= self.conv3_1(x)
        x= self.conv3_2(x)
        x= self.conv3_3(x)
        x= self.maxpool_3(x)


        #################Conv4#################
        x= self.conv4_1(x)
        x= self.conv4_2(x)
        x= self.conv4_3(x)
        x= self.maxpool_4(x)


        ################Conv5#################
        x= self.conv5_1(x)
        x= self.conv5_2(x)
        x= self.conv5_3(x)
        x= self.maxpool_5(x)

        return x





if __name__ == '__main__':

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(1, 3, 600, 1024)
    model = VGG()
    print(model.a)
    y_pred = model(x)
    print(y_pred)


