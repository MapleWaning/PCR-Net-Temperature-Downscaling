import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BasicUNet(nn.Module):
    def __init__(self, in_channels=20, out_channels=3, input_channels=None, output_channels=None):
        super().__init__()
        if input_channels is not None:
            in_channels = input_channels
        if output_channels is not None:
            out_channels = output_channels

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        def upsample_block(in_c):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c // 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_c // 2),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.center = conv_block(512, 1024)
        self.up4 = upsample_block(1024)
        self.dec4 = conv_block(1024, 512)
        self.up3 = upsample_block(512)
        self.dec3 = conv_block(512, 256)
        self.up2 = upsample_block(256)
        self.dec2 = conv_block(256, 128)
        self.up1 = upsample_block(128)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        base = x[:, 0:3, :, :]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        center = self.center(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(center), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return base + self.final(d1)


class SFTLayer(nn.Module):
    def __init__(self, feature_channels, prior_channels=1):
        super().__init__()
        self.prior_net = nn.Sequential(
            nn.Conv2d(prior_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.scale = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, feature_channels, kernel_size=3, padding=1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, feature_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, prior):
        if prior.shape[2:] != x.shape[2:]:
            prior = F.interpolate(prior, size=x.shape[2:], mode="bilinear", align_corners=True)
        prior_features = self.prior_net(prior)
        return x * (self.scale(prior_features) + 1.0) + self.shift(prior_features)


class AdaptiveTerrainAttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, kernel_size=1), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, kernel_size=1), nn.BatchNorm2d(f_int))
        self.w_dem = nn.Sequential(nn.Conv2d(1, f_int, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(f_int))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, dem):
        dem = F.interpolate(dem, size=g.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(self.w_g(g) + self.w_x(x) + self.alpha * self.w_dem(dem))
        return x * self.psi(psi)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True, use_sft=True):
        super().__init__()
        self.use_attention = use_attention
        self.use_sft = use_sft
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )
        if use_attention:
            self.attention = AdaptiveTerrainAttentionGate(in_channels // 2, skip_channels, skip_channels // 2)
        if use_sft:
            self.sft = SFTLayer(skip_channels, prior_channels=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip, dem=None):
        x = self.up(x)
        if self.use_sft and dem is not None:
            skip = self.sft(skip, dem)
        if self.use_attention and dem is not None:
            skip = self.attention(x, skip, dem)
        return self.conv(torch.cat([x, skip], dim=1))


class ResAttUNet(nn.Module):
    def __init__(
        self,
        in_channels=20,
        out_channels=3,
        use_attention=True,
        use_sft=True,
        dem_channel_index=3,
        pretrained_backbone=True,
    ):
        super().__init__()
        self.dem_index = dem_channel_index
        weights = models.ResNet34_Weights.DEFAULT if pretrained_backbone else None
        self.backbone = models.resnet34(weights=weights)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if weights is not None:
            with torch.no_grad():
                self.backbone.conv1.weight[:, :3] = original_conv1.weight

        self.enc1 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu)
        self.pool = self.backbone.maxpool
        self.enc2 = self.backbone.layer1
        self.enc3 = self.backbone.layer2
        self.enc4 = self.backbone.layer3
        self.center = self.backbone.layer4
        self.dec4 = DecoderBlock(512, 256, 256, use_attention, use_sft)
        self.dec3 = DecoderBlock(256, 128, 128, use_attention, use_sft)
        self.dec2 = DecoderBlock(128, 64, 64, use_attention, use_sft)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        base = x[:, 0:3, :, :]
        dem = x[:, self.dem_index : self.dem_index + 1, :, :]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        center = self.center(e4)

        d4 = self.dec4(center, e4, dem)
        d3 = self.dec3(d4, e3, dem)
        d2 = self.dec2(d3, e2, dem)
        d1 = self.dec1(d2, e1, dem)

        # The network learns a local residual over the normalized temperature base.
        return base + self.final_conv(self.final_up(d1))
