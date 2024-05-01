from torchvision.models.inception import inception_v3, Inception_V3_Weights



import torch.nn as nn
import torch
import os
import torchvision.models as models


class SketchNet(nn.Module):
    def __init__(self, pretrained=True) -> None:
        super().__init__()
        self.NUM_CLASSES = 250
        self.base = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.base.dropout = nn.Dropout(0.6)
        self.base.fc = nn.Linear(2048, self.NUM_CLASSES)

    def count_trainables(self):
        return sum(p.numel() for p in self.base.parameters() if p.requires_grad)

    def forward(self, x):
        return self.base(x)

if __name__ == "__main__":
    print(SketchNet())