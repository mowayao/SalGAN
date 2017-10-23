import numpy as np
from skimage.transform import rotate
from skimage.filters import gaussian_filter
import torch

__author__ = "ZEPING YAO"


class CenterCrop(object):
    def __call__(self, x, crop_size):#x's shape is [n, m, 3]
        assert x.ndim == 3
        centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
        halfw, halfh = crop_size[0] // 2, crop_size[1] // 2
        assert halfh <= centerh and halfw <= centerw
        return x[centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh]

class RandomCrop(object):
    def __call__(self, x, crop_size):## x's shape is [n, m, 3]
        assert x.ndim == 3
        assert x.shape[0] > crop_size[0] and x.shape[1] > crop_size[1]
        startw = np.random.randint(0, x.shape[0]-crop_size[0])
        starth = np.random.randint(0, x.shape[1]-crop_size[1])
        return x[startw: startw+crop_size[0], starth: starth+crop_size[1]]


class ToTensor(object):
    def __call__(self, x):
        assert x.ndim == 3
        x = x.transpose((2, 0, 1))
        return torch.from_numpy(x).float()

class ToImage(object):
    def __call__(self, x):
        return x.cpu().data.numpy().transpose((1,2,0))

class AddGaussianNoise(object):
    def __call__(self, x, mean, sigma):
        row, col, ch = x.shape
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        x += gauss
        return x

class GaussianBlurring(object):
    def __call__(self, x, sigma):
        image = gaussian_filter(x, sigma=(sigma, sigma, 0))
        return image

class Rotate(object):
	def __init__(self, angs, mode='reflect'):
		self.angs = angs
		self.mode = mode
	def __call__(self, x):
		angle = self.angs[np.random.choice(len(self.angs), 1)[0]]
		mi, ma = x.min(), x.max()
		x = rotate(x, angle, mode=self.mode, clip=True)
		return np.clip(x, mi, ma)
class RandomRotate(object):
    def __call__(self, x, max_ang, mode="reflect"):
        assert max_ang > 0
        angle = np.random.randint(-max_ang, max_ang)
        mi, ma = x.min(), x.max()
        x = rotate(x, angle, mode=mode, clip=True)
        return np.clip(x, mi, ma)

class Normalize(object):
    """Normalize each channel of the numpy array i.e.
    channel = (channel - mean) / std
    """
    def __call__(self, image):
        image /= 255.
        image -= image.mean(axis=(0, 1))
        s = image.std(axis=(0, 1))
        s[s == 0] = 1.0
        image /= s
        return image