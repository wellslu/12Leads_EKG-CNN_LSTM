import mlconfig
from torch import nn
import torch.nn.functional as F
import torch

class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ConvBNReLU, self).__init__(*layers)


class DwConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        layers = [
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels, padding=3, bias=False),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(DwConvBNReLU, self).__init__(*layers)


class Block_Layer(nn.Module):

    def __init__(self):
        super(Block_Layer, self).__init__()

        self.stems = nn.Sequential(
            DwConvBNReLU(12, 12, kernel_size=7, stride=2),
            ConvBNReLU(12, 32, kernel_size=7, stride=2),
            ConvBNReLU(32, 64, kernel_size=7, stride=2),
#             ConvBNReLU(64, 128, kernel_size=7, stride=2),
#            nn.AdaptiveMaxPool2d((1, 5)),
        )

        self.pool = nn.AdaptiveMaxPool1d(10)

    def forward(self, x):
        x = x
        x = self.stems(x)
        x = self.pool(x)
        return x.reshape(x.size(0), -1)
        
@mlconfig.register
class CNN_LSTM(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = Block_Layer()
        self.block2 = Block_Layer()
        
        self.classifier = nn.Sequential(
            nn.Linear(1780, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.embedding = nn.Embedding(100, 500)
        
        self._initialize_weights()

    def forward(self, x, bef):
        x = x.permute(3, 0, 1, 2)
        x1 = x[:12].permute(1, 0, 2, 3).reshape(x.size(1), 12, 1000)
        x2 = x[12:].permute(1, 0, 2, 3).reshape(x.size(1), 12, 1000)
        x1 = self.block1(x1)
        x2 = self.block2(x2)
#         x = x1 + x2
        x = torch.cat([x1, x2], 1)
        x = torch.cat([x, self.embedding(bef.long())], 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
