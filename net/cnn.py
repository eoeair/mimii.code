import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, channels, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # 修改点
        self.deconv1 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3)
        self.deconv2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3)
        self.deconv3 = nn.ConvTranspose1d(in_channels=32, out_channels=channels, kernel_size=3)

class CNN(nn.Module):
    def __init__(self, channels, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.deconv1 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=15, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=15, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose1d(in_channels=32, out_channels=13, kernel_size=13, stride=2, padding=2)


    def forward(self, x):
        # print("Input shape:", x.shape)
        x = F.relu(self.conv1(x))
        # print("After conv1 shape:", x.shape)
        x = F.relu(self.conv2(x))
        # print("After conv2 shape:", x.shape)
        x = F.relu(self.conv3(x))
        # print("After conv3 shape:", x.shape)

        x = self.global_avg_pool(x)
        # print("After global_avg_pool shape:", x.shape)

        x = x.expand(-1, -1, 10)
        # print("After expand shape:", x.shape)

        x = F.relu(self.deconv1(x))
        # print("After deconv1 shape:", x.shape)
        x = F.relu(self.deconv2(x))
        # print("After deconv2 shape:", x.shape)
        x = F.relu(self.deconv3(x))
        # print("After deconv3 shape:", x.shape)

        return x

