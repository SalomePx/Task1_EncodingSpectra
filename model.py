import torch.nn as nn
import torch

class EncoderBlock(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        outputs = self.relu(x)
        return outputs
    
class FTIREncoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.s1 = nn.Sequential(
            EncoderBlock(in_channels, 16),
            nn.Conv1d(16, 16, kernel_size=3, stride=2),
        )

        self.s2 = nn.Sequential(
            EncoderBlock(16, 32),
            nn.Conv1d(32, 32, kernel_size=3, stride=2),
        )

        self.s3 = nn.Sequential(
            EncoderBlock(32, 64), 
            nn.Conv1d(64, 64, kernel_size=3, stride=2)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64, latent_dim)
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, inputs):
        x = self.s1(inputs)
        x = self.s2(x)
        x = self.s3(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        outputs = self.head(x)
        return outputs

if __name__ == "__main__":
    spectrum = torch.randn(8, 1, 4000)
    model = FTIREncoder(1, 100)
    print(model(spectrum).shape)