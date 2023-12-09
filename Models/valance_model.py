import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.BasicModule import BasicModule
use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')

class DCD(BasicModule):
    def __init__(self, h_features=64, input_features=128):
        super(DCD, self).__init__()
        self.fc1 = nn.Linear(input_features, h_features)
        self.fc2 = nn.Linear(h_features, h_features)
        self.fc3 = nn.Linear(h_features, 6)

    def forward(self, inputs):
        out = F.relu(self.fc1(inputs))
        out = self.fc2(out)
        return F.softmax(self.fc3(out), dim=1)


class Classifier(BasicModule):
    def __init__(self, input_features=64):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_features, 2)  #分类种类

    def forward(self, input):
        return F.softmax(self.fc(input), dim=1)


class Encoder(BasicModule):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.conv1=nn.Conv2d(1, 6, 5)
        # self.conv2=nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(128, 124)
        self.fc2 = nn.Linear(124, 84)
        self.fc3 = nn.Linear(84, 64)
        self.dt = nn.Dropout(0.5)

    def forward(self, input):
        input = input.float()
        # input = torch.tensor(input)
        # out=F.relu(self.conv1(input))
        # out=F.max_pool2d(out,2)
        # out=F.relu(self.conv2(out))
        # out=F.max_pool2d(out,2)
        # out=out.view(out.size(0),-1)

        out = F.relu(self.fc1(input))
        # out = self.dt(out)
        out = F.relu(self.fc2(out))
        # out = self.dt(out)
        out = self.fc3(out)
        # out = self.dt(out)

        return out