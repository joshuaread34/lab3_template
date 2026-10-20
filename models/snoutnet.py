"""
ELEC 475 Lab 3: Pet Nose Localization Models
"""

import torch
import torch.nn as nn
from torchvision.models import alexnet, vgg16, AlexNet_Weights, VGG16_Weights


class SnoutNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SnoutNetAlexNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = AlexNet_Weights.DEFAULT if pretrained else None
        self.backbone = alexnet(weights=weights)
        self.backbone.classifier[6] = nn.Linear(4096, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SnoutNetVGG16(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.backbone = vgg16(weights=weights)
        self.backbone.classifier[6] = nn.Linear(4096, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    from torchinfo import summary

    print("SnoutNet summary:")
    summary(SnoutNet(), input_size=(1, 3, 227, 227))

    models = {
        "SnoutNet": SnoutNet(),
        "SnoutNetAlexNet": SnoutNetAlexNet(pretrained=False),
        "SnoutNetVGG16": SnoutNetVGG16(pretrained=False),
    }

    x = torch.randn(1, 3, 227, 227)

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(x)

        params = sum(p.numel() for p in model.parameters())
        print(
            f"{name}: output shape={tuple(output.shape)}, "
            f"parameters={params:,}"
        )

    print("\nLoading provided checkpoints:")
    checks = [
        (SnoutNet(), "snoutnet.pth"),
        (SnoutNetAlexNet(pretrained=False), "alexnet.pth"),
        (SnoutNetVGG16(pretrained=False), "vgg16.pth"),
    ]

    for model, name in checks:
        state = torch.load(
            f"checkpoints/{name}",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        print(f"Loaded {name}")

    from utils.dataset import PetNoseDataset

    model = checks[0][0]
    dataset = PetNoseDataset(split="test")
    model.eval()

    image, label, orig_size = dataset[0]
    with torch.no_grad():
        pred = model(image.unsqueeze(0))[0]

    print("Ground truth:", label / 227 * orig_size)
    print("Prediction:  ", pred * orig_size)
