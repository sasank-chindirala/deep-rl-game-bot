import torch
import torch.nn as nn
import torch.nn.functional as F


class Generalized_CNN(nn.Module):
    def __init__(self):
        super(Generalized_CNN, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.fc1 = nn.Linear(32 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 9)
        self.double()

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


