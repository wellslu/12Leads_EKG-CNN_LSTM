import mlconfig
from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,kernel_size), stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ConvBNReLU, self).__init__(*layers)
        
class Block_Layer(nn.Module):

    def __init__(self):
        super(Block_Layer, self).__init__()
        self.feature_dim = 1
        self.hidden_dim = 64
        self.num_layers = 3
        self.output_dim = 128
        self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim, self.num_layers, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim , self.output_dim)
        
        self.cnn = nn.Sequential(
            ConvBNReLU(1, 16),
            ConvBNReLU(16, 32),
            nn.MaxPool2d((1,7),1),
            ConvBNReLU(32, 64),
            ConvBNReLU(64, 128),
            nn.AdaptiveMaxPool2d((1, 1)),
        )

        self.attent =  nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
        )

    def forward(self, x):
        h0 = torch.zeros([self.num_layers, x.size(0), self.hidden_dim], dtype=torch.float).to(device)
        c0 = torch.zeros([self.num_layers, x.size(0), self.hidden_dim], dtype=torch.float).to(device)
        x_lstm = x.permute(0, 2, 1)
        x_lstm, _ = self.lstm(x_lstm, (h0.detach(), c0.detach()))
        x_lstm = self.fc(x_lstm[:, -1, :])
        
        x_cnn = x.reshape(x.size(0), 1, 1, -1)
        x_cnn = self.cnn(x_cnn)
        
        x = torch.cat([x_lstm.reshape(x.size(0), -1), x_cnn.reshape(x.size(0), -1)], 1)
        x = self.attent(x)
        return x
        
@mlconfig.register
class CNN_LSTM2(nn.Module):

    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.block1 = Block_Layer()
        self.block2 = Block_Layer()
        self.block3 = Block_Layer()
        self.block4 = Block_Layer()
        self.block5 = Block_Layer()
        self.block6 = Block_Layer()
        self.block7 = Block_Layer()
        self.block8 = Block_Layer()
        self.block9 = Block_Layer()
        self.block10 = Block_Layer()
        self.block11 = Block_Layer()
        self.block12 = Block_Layer()
        
        self.classifier = nn.Sequential(
            nn.Linear(192, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            # nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
            # nn.Softmax(1),
        )
        
        self._initialize_weights()

    def forward(self, x):
        x = x.permute(3, 0, 1, 2)
        outs = []
        outs.append(self.block1(x[0]))
        outs.append(self.block2(x[1]))
        outs.append(self.block3(x[2]))
        outs.append(self.block4(x[3]))
        outs.append(self.block5(x[4]))
        outs.append(self.block6(x[5]))
        outs.append(self.block7(x[6]))
        outs.append(self.block8(x[7]))
        outs.append(self.block9(x[8]))
        outs.append(self.block10(x[9]))
        outs.append(self.block11(x[10]))
        outs.append(self.block12(x[11]))
        x = outs[0]
        for out in outs[1:]:
            # x = x + out
            x = torch.cat([x, out], 1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
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
