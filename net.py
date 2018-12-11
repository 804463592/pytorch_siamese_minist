# -*- encoding: utf-8 -*-
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),   # input_shape:(,1,28,28),output:(20,24,24)
            nn.MaxPool2d(2, stride=2),         # output:(20,12,12)
            nn.Conv2d(20, 50, kernel_size=5),  # output:(50,8,8)
            nn.MaxPool2d(2, stride=2))     # output:(50,4,4)

        self.fc1 = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            nn.Linear(10, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
