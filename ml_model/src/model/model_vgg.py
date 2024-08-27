import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CustomVGG(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomVGG, self).__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(base_model.classifier[0].in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def initialize_vgg16(num_classes):
    vgg16_base = models.vgg16(pretrained=True)
    for param in vgg16_base.parameters():
        param.requires_grad = False
    return CustomVGG(vgg16_base, num_classes)

def initialize_vgg19(num_classes):
    vgg19_base = models.vgg19(pretrained=True)
    for param in vgg19_base.parameters():
        param.requires_grad = False
    return CustomVGG(vgg19_base, num_classes)