import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


# Load pretrained MobileNetV3
# weights="IMAGENET1K_V1" loads the model trained on ImageNet
model = mobilenet_v3_small(weights="IMAGENET1K_V1")


num_classes = 5
# The last linear layer is index 3 in the classifier
model.classifier[3] = nn.Linear(in_features=1024, out_features=num_classes)

print(model)


