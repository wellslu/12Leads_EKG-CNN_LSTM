import mlconfig
from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@mlconfig.register
class LSTMModel(nn.Module):

    def __init__(self):
        super(LSTMModel, self).__init__()
        self.feature_dim = 1
        self.hidden_dim = 64
        self.num_layers = 3
        self.output_dim = 128
        
        self.lstm1 = nn.LSTM(self.feature_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim , self.output_dim)
        self.lstm2 = nn.LSTM(self.feature_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc2 = nn.Linear(self.hidden_dim , self.output_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(129, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        self._initialize_weights()

    def forward(self, x, bef):
        h01 = torch.zeros([self.num_layers, x.size(0), self.hidden_dim], dtype=torch.float).to(device)
        c01 = torch.zeros([self.num_layers, x.size(0), self.hidden_dim], dtype=torch.float).to(device)
        h02 = torch.zeros([self.num_layers, x.size(0), self.hidden_dim], dtype=torch.float).to(device)
        c02 = torch.zeros([self.num_layers, x.size(0), self.hidden_dim], dtype=torch.float).to(device)
        x = x.permute(3, 0, 1, 2)
        x1 = x[:12].permute(1, 3, 2, 0)
        x2 = x[12:].permute(1, 3, 2, 0)
        
        x1, _ = self.lstm1(x1, (h01.detach(), c01.detach()))
        x1 = self.fc1(x1[:, -1, :]).reshape(x.size(1), -1)
        x2, _ = self.lstm2(x2, (h02.detach(), c02.detach()))
        x2 = self.fc2(x2[:, -1, :]).reshape(x.size(1), -1)
        x = x1 - x2
        x = torch.cat([x, bef.reshape(-1, 1)], 1)
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