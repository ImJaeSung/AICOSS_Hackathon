import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# Weights ported from https://github.com/lukemelas/EfficientNet-PyTorch/

"""https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth"""
class Efficientb4(nn.Module):
    def __init__(self, num_classes = 60, T = 1):
        super(Efficientb4, self).__init__()
        self.backbone = models.efficientnet_b4(pretrained = True)
        self.classifier = nn.Sequential(
                nn.Linear(1000, 500), 
                nn.ReLU(),           
                nn.Linear(500, 250),
                nn.ReLU(),
                nn.Linear(250, num_classes)
                )
        self.T = T
        
    def temperature_sigmoid(self, x, T):
        return 1 / (1 + torch.exp(-x / T))
                                
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)

        return self.temperature_sigmoid(x, self.T)
    
"""https://download.pytorch.org/models/efficientnet_b7_lukemelas-c5b4e57e.pth"""
class Efficientb7(nn.Module):
    def __init__(self, num_classes = 60, T = 1):
        super(Efficientb7, self).__init__()
        self.backbone = models.efficientnet_b7(pretrained = True)
        self.classifier = nn.Sequential(
                nn.Linear(1000, 512), 
                nn.ReLU(),
                nn.Dropout(0.2),  # Dropout 추가            
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),  # Dropout 추가
                nn.Linear(256, num_classes)
                )
        self.T = T
        
    def temperature_sigmoid(self, x, T):
        return 1 / (1 + torch.exp(-x / T))
                                
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)

        return self.temperature_sigmoid(x, self.T)