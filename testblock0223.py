import torch.nn as nn
from unet import UNet

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=1, padding=1, bias=True)


def conv2x2(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=2,
                     stride=2, padding=0, bias=True)


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=1, padding=0, bias=True)

class network1(nn.Module):
    def __init__(self):
        super(network1, self).__init__()
        
        self.segnet = UNet()

    def forward(self, x1,   future_seq=0, hidden_state=None):

        shape1 = self.segnet(x1)

        return shape1