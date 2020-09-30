from typing import Optional
import torch
import torch.nn as nn
from torch.autograd import Variable
from segmentation_models_pytorch.base import ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import initialization as init


class Classifier(nn.Module):
    def __init__(self, num_classes, input_shape=64):
        super().__init__()
        self.fc = nn.Linear(input_shape, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class UnetClassifier(torch.nn.Module):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 in_channels: int = 3,
                 classes: int = 1,
                 aux_params: Optional[dict] = None,
                 ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.num_classes = classes
        self.classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1], **aux_params
        )
        self.name = "c-{}".format(encoder_name)
        init.initialize_head(self.classification_head)

    def forward(self, x):
        ### Pass model through the encoder and the classifier part of it.
        features = self.encoder(x)
        labels = self.classification_head(features[-1])
        # -- For compatibility with the rest of the code. Output a zero mask region.
        return Variable(torch.zeros(*x.shape)), labels


