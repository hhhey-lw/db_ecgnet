import torch
from torch import nn


# Test whether the program is running normally
class SimpleNet(nn.Module):
    def __init__(self, input_channels=12, num_classes=9):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_channels),
            nn.Conv1d(input_channels, 16, kernel_size=100, stride=100),
            nn.GELU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)


if __name__ == '__main__':
    x = torch.randn((1, 12, 1000))
    f = SimpleNet().eval()
    y = f(x)
    print(y.shape)
