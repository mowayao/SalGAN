import torch
import torch.nn as nn
import torch.nn.functional as F



class Decoder_UNet(nn.Module):
    def __init__(self, input_dim, features, output_dim):
        super(Decoder_UNet, self).__init__()
        layers = [
            nn.Conv2d(input_dim, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        ]
        self.down = nn.Sequential(*layers)
    def forward(self, x):
        return self.down(x)

class Encoder_UNet(nn.Module):
    def __init__(self, in_channels, features, out_channels):
        super(Encoder_UNet, self).__init__()
        layers = [
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.up = nn.Sequential(*layers)
    def forward(self, x):
        return self.up(x)
class Center_UNet(nn.Module):
    def __init__(self, in_channels, features, out_channels):
        super(Center_UNet, self).__init__()
        layers = [
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.cen = nn.Sequential(*layers)
    def forward(self, x):
        return self.cen(x)


class UNet1024(nn.Module):
    def __init__(self, num_classes):
        super(UNet1024, self).__init__()
        self.dec1 = Decoder_UNet(3, 8, 8)
        self.dec2 = Decoder_UNet(8, 16, 16)
        self.dec3 = Decoder_UNet(16, 32, 32)
        self.dec4 = Decoder_UNet(32, 64, 64)
        self.dec5 = Decoder_UNet(64, 128, 128)
        self.dec6 = Decoder_UNet(128, 256, 256)
        self.dec7 = Decoder_UNet(256, 512, 512)
        self.center = Center_UNet(512, 1024, 1024)
        self.enc7 = Encoder_UNet(1024+512, 512, 512)
        self.enc6 = Encoder_UNet(512+256, 256, 256)
        self.enc5 = Encoder_UNet(256+128, 128, 128)
        self.enc4 = Encoder_UNet(128+64, 64, 64)
        self.enc3 = Encoder_UNet(64+32, 32, 32)
        self.enc2 = Encoder_UNet(32+16, 16, 16)
        self.enc1 = Encoder_UNet(16+8, 8, 8)

        self.classify = nn.Conv2d(8, num_classes, 1)

    def forward(self, x):
        dec1 = self.dec1(x)
        out = F.max_pool2d(dec1, kernel_size=2, stride=2)
        dec2 = self.dec2(out)
        out = F.max_pool2d(dec2, kernel_size=2, stride=2)
        dec3 = self.dec3(out)
        out = F.max_pool2d(dec3, kernel_size=2, stride=2)
        dec4 = self.dec4(out)
        out = F.max_pool2d(dec4, kernel_size=2, stride=2)
        dec5 = self.dec5(out)
        out = F.max_pool2d(dec5, kernel_size=2, stride=2)
        dec6 = self.dec6(out)
        out = F.max_pool2d(dec6, kernel_size=2, stride=2)
        dec7 = self.dec7(out)
        out = F.max_pool2d(dec7, kernel_size=2, stride=2)
        out = self.center(out)
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc7(torch.cat([dec7, out], dim=1))

        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc6(torch.cat([dec6, out], dim=1))
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc5(torch.cat([dec5, out], dim=1))
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc4(torch.cat([dec4, out], dim=1))
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc3(torch.cat([dec3, out], dim=1))
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc2(torch.cat([dec2, out], dim=1))
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc1(torch.cat([dec1, out], dim=1))
        out = torch.squeeze(out, dim=1)
        return self.classify(out)

class UNet512(nn.Module):
    def __init__(self, num_classes):
        super(UNet512, self).__init__()
        self.dec1 = Decoder_UNet(3, 16, 16)
        self.dec2 = Decoder_UNet(16, 32, 32)
        self.dec3 = Decoder_UNet(32, 64, 64)
        self.dec4 = Decoder_UNet(64, 128, 128)
        self.dec5 = Decoder_UNet(128, 256, 256)
        self.dec6 = Decoder_UNet(256, 512, 512)
        self.center  =  Center_UNet(512, 1024, 1024)
        self.enc6 = Encoder_UNet(512+1024, 512, 512)
        self.enc5 = Encoder_UNet(512+256, 256, 256)
        self.enc4 = Encoder_UNet(256+128, 128, 128)
        self.enc3 = Encoder_UNet(128+64, 64, 64)
        self.enc2 = Encoder_UNet(64+32, 32, 32)
        self.enc1 = Encoder_UNet(32+16, 16, 16)
        self.classify = nn.Conv2d(16, num_classes, 1, 1, 0)

    def forward(self, x):
        dec1 = self.dec1(x)
        out = F.max_pool2d(dec1, kernel_size=2, stride=2)
        dec2 = self.dec2(out)
        out = F.max_pool2d(dec2, kernel_size=2, stride=2)
        dec3 = self.dec3(out)
        out = F.max_pool2d(dec3, kernel_size=2, stride=2)
        dec4 = self.dec4(out)
        out = F.max_pool2d(dec4, kernel_size=2, stride=2)
        dec5 = self.dec5(out)
        out = F.max_pool2d(dec5, kernel_size=2, stride=2)
        dec6 = self.dec6(out)
        out = F.max_pool2d(dec6, kernel_size=2, stride=2)
        out = self.center(out)
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc6(torch.cat([dec6, out], dim=1))
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc5(torch.cat([dec5, out], dim=1))
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc4(torch.cat([dec4, out], dim=1))
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc3(torch.cat([dec3, out], dim=1))
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc2(torch.cat([dec2, out], dim=1))
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = self.enc1(torch.cat([dec1, out], dim=1))
        return self.classify(out)

class UNet256(nn.Module):
    def __init__(self, num_classes=1):
        super(UNet256, self).__init__()
        self.dec1 = Decoder_UNet(3, 16, 32)
        self.dec2 = Decoder_UNet(32, 64, 128)
        self.dec3 = Decoder_UNet(128, 256, 512)
        self.dec4 = Decoder_UNet(512, 512, 512)
        self.cen = Center_UNet(512, 512)
        self.enc4 = Encoder_UNet(1024, 512, 512)
        self.enc3 = Encoder_UNet(1024, 512, 128)
        self.enc2 = Encoder_UNet(256, 128, 32)
        self.enc1 = Encoder_UNet(64, 32, 32)
        self.classify = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        dec1 = self.dec1(x)
        out = F.max_pool2d(dec1, 2, 2)
        dec2 = self.dec2(out)
        out = F.max_pool2d(dec2, 2, 2)
        dec3 = self.dec3(out)
        out = F.max_pool2d(dec3, 2, 2)
        dec4 = self.dec4(out)
        out = F.max_pool2d(dec4, 2, 2)
        out = self.cen(out)

        out = self.enc4(torch.cat([dec4, F.upsample(out, scale_factor=2, mode='bilinear')], dim=1))
        out = self.enc3(torch.cat([dec3, F.upsample(out, scale_factor=2, mode='bilinear')], dim=1))
        out = self.enc2(torch.cat([dec2, F.upsample(out, scale_factor=2, mode='bilinear')], dim=1))
        out = self.enc1(torch.cat([dec1, F.upsample(out, scale_factor=2, mode='bilinear')], dim=1))
        return self.classify(out)
#unet = UNet256(1)