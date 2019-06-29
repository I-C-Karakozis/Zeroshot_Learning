import math

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

def load_to_device(net):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    if device == 'cuda':
        cudnn.benchmark = True
    return net, device

# ---------------------------- ResNet ---------------------------- #

'''
Reference:
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

def initLinear(linear, val=None):
    if val is None:
        fan = linear.in_features + linear.out_features
        spread = math.sqrt(2.0) * math.sqrt(2.0/fan)
    else:
        spread = val
    linear.weight.data.uniform_(-spread, spread)
    linear.bias.data.uniform_(-spread, spread)
    return

class ResNet18(nn.Module):
    def base_size(self): return 512

    def __init__(self, n_classes, pretrained):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.n_classes = n_classes

        # get resnet layers
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # define classification layers
        self.dropout2d = nn.Dropout2d(.5)
        self.linear = nn.Linear(7*7*self.base_size(), self.n_classes)
        initLinear(self.linear)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
     
        x = self.dropout2d(x)
        cls_scores = self.linear(x.view(-1, 7*7*self.base_size()))
        return cls_scores

def load_resnet18(n_classes, pretrained=False):
    net = ResNet18(n_classes, pretrained=pretrained)
    return load_to_device(net) 

class ResNet34(nn.Module):
    def base_size(self): return 512
    def rep_size(self): return 1024

    def __init__(self, n_classes, pretrained):
        super(ResNet34, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        self.n_classes = n_classes

        # get resnet layers
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # define layers
        self.linear1 = nn.Linear(7 * 7 * self.base_size(), self.rep_size())
        self.linear2 = nn.Linear(self.rep_size(), self.n_classes)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()

        # initialize linear layers
        initLinear(self.linear1)
        initLinear(self.linear2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
     
        x = self.dropout2d(x)
        x = self.relu(self.linear1(x.view(-1, 7*7*self.base_size())))
        x = self.dropout(x.clone())
        cls_scores = self.linear2(x.clone())

        return cls_scores

def load_resnet34(n_classes, pretrained=False):
    net = ResNet34(n_classes, pretrained=pretrained)
    return load_to_device(net)   

class ResNet50(nn.Module):
    def base_size(self): return 2048
    def rep_size(self): return 1024
    
    def __init__(self, n_classes, pretrained):
        super(ResNet50, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.n_classes = n_classes

        # get resnet layers
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # define layers
        self.linear1 = nn.Linear(7 * 7 * self.base_size(), self.rep_size())
        self.linear2 = nn.Linear(self.rep_size(), self.n_classes)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()

        # initialize linear layers
        initLinear(self.linear1)
        initLinear(self.linear2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
     
        x = self.dropout2d(x)
        x = self.relu(self.linear1(x.view(-1, 7*7*self.base_size())))
        x = self.dropout(x.clone())
        cls_scores = self.linear2(x.clone())

        return cls_scores

def load_resnet50(n_classes, pretrained=False):
    net = ResNet50(n_classes, pretrained=pretrained)
    return load_to_device(net) 
