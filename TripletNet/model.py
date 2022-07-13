import torch
import torch.nn as nn

from torchvision.models import resnet50

class TripletNet(nn.Module):
    def __init__(self, embedding_size=64):
        super(TripletNet, self).__init__()
        resnet = resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = True
        resnet.fc = nn.Linear(2048, embedding_size)
        
        self.resnet = resnet

    def forward(self, imgs):
        preds = self.resnet(imgs)
        return preds 