import torch
import torch.nn as nn

from torchvision.models import resnet50

class TripletNet(nn.Module):
    def __init__(self, n_classes=81):
        super(TripletNet, self).__init__()
        resnet = resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = True
        resnet.fc = nn.Linear(2048, n_classes)
        
        self.resnet = resnet
        self.sfmx = nn.Softmax(dim=1)

    def forward(self, anc_img, pos_img, neg_img):
        anc_pred = self.sfmx(self.resnet(anc_img))
        pos_pred = self.sfmx(self.resnet(pos_img))
        neg_pred = self.sfmx(self.resnet(neg_img))

        return anc_pred, pos_pred, neg_pred 

    def predict(self, img):
        pred = self.sfmx(self.resnet(img))
        return pred