import numpy as np
import torch
import torch.nn as nn
from utils.move_index import NUM_MOVES
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return self.relu(out)

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, 3)
        self.res1 = ResBlock(128)

        self.conv3 = nn.Conv2d(128, 128, 3)
        self.res2 = ResBlock(128)

        self.conv4 = nn.Conv2d(128, 256, 3)
        self.res3 = ResBlock(256)

        flat = 256 * 2 * 2

        #Value Head
        self.value_fc1 = nn.Linear(flat, 512)  
        self.value_fc2 = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        
        #Policy Head
        self.num_moves = NUM_MOVES
        self.policy_f1 = nn.Linear(flat, 1024)
        self.policy_f2 = nn.Linear(1024, self.num_moves)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.res1(x)

        x = self.relu(self.conv3(x))
        x = self.res2(x)
        
        x = self.relu(self.conv4(x))
        x = self.res3(x)

        x = x.view(x.size(0), -1)
    
        v = self.relu(self.value_fc1(x))
        v = self.tanh(self.value_fc2(v))

        p = self.relu(self.policy_f1(x))
        p = self.policy_f2(p)
        
        return p, v
