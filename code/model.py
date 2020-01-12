import torch
import torch.nn as nn
import torch.nn.functional as F

class SPCH2FLM(nn.Module):
    def __init__(self, numFilters=64, filterWidth=21):
        super(SPCH2FLM, self).__init__()
        self.numFilters = numFilters
        self.filterWidth = filterWidth
        self.conv1 = nn.Conv1d(1, self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1)
        self.conv2 = nn.Conv1d(self.numFilters, 2*self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1)     
        self.conv3 = nn.Conv1d(2*self.numFilters, 4*self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1)
        self.conv4 = nn.Conv1d(4*self.numFilters, 8*self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1)
        self.fc1 = nn.Linear(62464, 6) 

    def forward(self, x):
        h = F.dropout(F.leaky_relu(self.conv1(x), 0.3), 0.2)
        h = F.dropout(F.leaky_relu(self.conv2(h), 0.3), 0.2)
        h = F.dropout(F.leaky_relu(self.conv3(h), 0.3), 0.2)
        h = F.dropout(F.leaky_relu(self.conv4(h), 0.3), 0.2)
        features = h = h.view(h.size(0), -1)
        h = F.leaky_relu(self.fc1(h), 0.3)
        return h, features

class SPCH2FLMTC(nn.Module):
    def __init__(self, numFilters=64, filterWidth=21):
        super(SPCH2FLMTC, self).__init__()
        self.numFilters = numFilters
        self.filterWidth = filterWidth
        self.conv1 = nn.Conv1d(1, self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1)
        self.conv2 = nn.Conv1d(self.numFilters, 2*self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1)     
        self.conv3 = nn.Conv1d(2*self.numFilters, 4*self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1)
        self.conv4 = nn.Conv1d(4*self.numFilters, 8*self.numFilters, self.filterWidth, stride=2, padding=0, dilation=1)
        self.fc1 = nn.Linear(62464, 6) # 1536
        self.fc2 = nn.Linear(12, 6)

    def forward(self, x, c):
        h = F.dropout(F.leaky_relu(self.conv1(x), 0.3), 0.2)
        h = F.dropout(F.leaky_relu(self.conv2(h), 0.3), 0.2)
        h = F.dropout(F.leaky_relu(self.conv3(h), 0.3), 0.2)
        h = F.dropout(F.leaky_relu(self.conv4(h), 0.3), 0.2)
        features = h = h.view(h.size(0), -1)
        h = F.leaky_relu(self.fc1(h), 0.3)
        h = torch.cat((h, c), -1)
        h = self.fc2(h)
        return h, features