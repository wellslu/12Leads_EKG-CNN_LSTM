import mlconfig
from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2, padding=1):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,kernel_size), stride=stride, bias=False),
#             nn.Conv2d(in_channels, out_channels, kernel_size=(1,kernel_size), stride=stride, groups=1000),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ConvBNReLU, self).__init__(*layers)
        
class Block_Layer(nn.Module):

    def __init__(self):
        super(Block_Layer, self).__init__()
        self.feature_dim = 1
        self.hidden_dim = 256
        self.num_layers = 3
        self.cn = ConvBNReLU(1000, 512, 12, 1)
        
        self.cnn = nn.Sequential(
            ConvBNReLU(12, 32),
            ConvBNReLU(32, 32, 3, 1),
            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64, 3, 1),
            ConvBNReLU(64, 128),
            nn.AdaptiveMaxPool2d((1, 1)),
        )

    def forward(self, x):
        
        x_cnn = x.permute(0, 3, 1, 2)
        x_cnn = self.cnn(x_cnn)
        
        x = x_cnn.reshape(x.size(0), -1)
        return x
        
@mlconfig.register
class CNN_LSTM(nn.Module):

    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.block = Block_Layer()
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        
        self._initialize_weights()

    def forward(self, x):
        x = self.block(x)
        x = self.classifier(x)
#         x = torch.sigmoid(x)
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