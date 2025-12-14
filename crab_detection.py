"""Detects crabs in images"""

import torch
from torchvision import models, transforms
from PIL import Image

device = torch.device("cpu")


def main():
    resnet50 = models.resnet50(pretrained=True)


if __name__ == "__main__":
    main()
