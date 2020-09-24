# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        input_size = 784
        output_size = 10
        self.main = nn.Linear(input_size,output_size)
    
    def forward(self, x):
        output = x.view(x.size(0), -1)
        output = self.main(output)
        return F.log_softmax(output)

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        input_size = 784
        hidden_sizes = 80
        output_size = 10
        self.layer_one = nn.Sequential(
                        nn.Linear(input_size, hidden_sizes),
                        nn.Tanh()
        )
        self.layer_two = nn.Sequential(
                        nn.Linear(hidden_sizes, output_size),
        )

    def forward(self, x):
        # CHANGE CODE HERE
        output = x.view(x.size(0), -1)
        output = self.layer_one(output)
        output = self.layer_two(output)
        output = F.log_softmax(output, dim = 1) 
        return output

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        input_size = 28
        hidden_sizes = [125, 250]
        output_size = 10

        self.layer1 = nn.Sequential(nn.Conv2d(1, hidden_sizes[0], kernel_size = 5, padding = 2, stride = 2 ),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(nn.Conv2d(hidden_sizes[0], hidden_sizes[1], kernel_size = 5, padding = 2, stride = 2),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=2, stride = 2)
        )
        self.drop_out = nn.Dropout()
        self.fullConn = nn.Sequential(nn.Linear(1000 ,100),
                        nn.ReLU(),
                        nn.Linear(100, 10),
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = output.view(output.size(0), -1)
        # output = self.drop_out(output)
        output = self.fullConn(output)
        return F.log_softmax(output)
        