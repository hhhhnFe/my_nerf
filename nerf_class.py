import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class NeRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(60, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(316, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 256)

        self.density = nn.Linear(256, 1)
        self.feature = nn.Linear(256, 256)
        self.fc9 = nn.Linear(280, 128)
        self.rgb = nn.Linear(128, 3)
        
    def forward(self, r_x, r_d):
        out = F.relu(self.fc1(r_x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))

        out = torch.cat([out, r_x], -1)
        out = F.relu(self.fc6(out))
        out = F.relu(self.fc7(out))
        out = F.relu(self.fc8(out))

        density = self.density(out) # ReLU applied later
        out = torch.cat([self.feature(out), r_d], -1)
        out = F.relu(self.fc9(out))
        out = self.rgb(out) # /sigmoid applied later

        return torch.cat([out, density], dim=-1)
