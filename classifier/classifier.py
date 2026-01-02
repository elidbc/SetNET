import torch
import torch.nn as nn
import torch.nn.functional as F


class SetClassifier(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=3, padding=1), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = 4*channels

        self.head_shape = nn.Linear(feat_dim, 3)
        self.head_color = nn.Linear(feat_dim, 3)
        self.head_shading = nn.Linear(feat_dim, 3)
        self.head_count = nn.Linear(feat_dim, 3)

    
    def forward(self, x):
        z = self.backbone(x)
        z = self.pool(z).flatten(1)
        return {
            "shape": self.head_shape(z),
            "color": self.head_color(z),
            "shading": self.head_shading(z),
            "count": self.head_count(z)
        }
        

