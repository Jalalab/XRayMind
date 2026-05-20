import torch
import torch.nn as nn
from torchvision import models

DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES = len(DISEASES)


class MediScanModel(nn.Module):
    def __init__(self, num_classes=14, dropout=0.4):
        super(MediScanModel, self).__init__()

        # DenseNet121 — same architecture as CheXNet (Stanford 2017)
        self.backbone = models.densenet121(weights=None)

        in_features = self.backbone.classifier.in_features  # 1024
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
