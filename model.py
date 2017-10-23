import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
class ConBNRelu(nn.Module):
	def __init__(self, in_channels, out_channels, k_size, stride, padding):
		super(ConBNRelu, self).__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, k_size, stride, padding),
			nn.ReLU()
		)
	def forward(self, x):
		return self.block(x)


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.block = nn.Sequential(
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.Conv2d(512, 512, 3, 1, 1),
			#nn.ConvTranspose2d(512, 512, 4, 2, 1),
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.Conv2d(512, 512, 3, 1, 1),
			#nn.ConvTranspose2d(512, 512, 4, 2, 1),
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(512, 256, 3, 1, 1),
			nn.Conv2d(256, 256, 3, 1, 1),
			nn.Conv2d(256, 256, 3, 1, 1),
			#nn.ConvTranspose2d(256, 256, 4, 2, 1),
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(256, 128, 3, 1, 1),
			nn.Conv2d(128, 128, 3, 1, 1),
			nn.Conv2d(128, 128, 3, 1, 1),
			#nn.ConvTranspose2d(128, 128, 4, 2, 1),
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(128, 64, 3, 1, 1),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.Conv2d(64, 1, 1, 1, 0),
			nn.Sigmoid()
		)

	def forward(self, x):
		return torch.squeeze(self.block(x))

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		vgg_bn = models.vgg16_bn(pretrained=True)
		params = list(vgg_bn.features.children())[:-1]
		#params = filter(lambda x: not isinstance(x, nn.BatchNorm2d), params)
		self.block = nn.Sequential(*params)
	def forward(self, x):
		return self.block(x)

class Discriminator(nn.Module):
	def __init__(self, IMG_SIZE):
		super(Discriminator, self).__init__()
		h, w = IMG_SIZE
		self.feats = nn.Sequential(
			ConBNRelu(4, 3, 1, 1, 0),
			ConBNRelu(3, 32, 3, 1, 1),
			nn.MaxPool2d(2, 2),
			ConBNRelu(32, 64, 3, 1, 1),
			ConBNRelu(64, 64, 3, 1, 1),
			nn.MaxPool2d(2, 2),
			ConBNRelu(64, 64, 3, 1, 1),
			ConBNRelu(64, 64, 3, 1, 1),
			nn.MaxPool2d(2, 2)
		)
		self.classifer = nn.Sequential(
			nn.Linear(64*h/8*w/8, 100),
			nn.Tanh(),
			nn.Linear(100, 2),
			nn.Tanh(),
			nn.Linear(2, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.feats(x)
		x = x.view(x.size(0), -1)
		x = self.classifer(x)
		return x

class SalGAN(nn.Module):
	def __init__(self):
		super(SalGAN, self).__init__()
		self.encoder = Encoder()
		self.decoder = Decoder()

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x
if __name__ == "__main__":
	pass

