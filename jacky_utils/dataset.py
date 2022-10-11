import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

class Normalization:
    def normalize(self, x):
        raise NotImplementedError()

    def invert_normalize(self, x):
        raise NotImplementedError()


class MeanStdNormalization(Normalization):
    def __init__(self, mean=127.0, std=73.6):
        self.mean = mean
        self.std = std

    def normalize(self, x):
        return x * self.std + self.mean

    def invert_normalize(self, x):
        return (x - self.mean) / self.std


class MultiChannnelMeanStdNormalization(Normalization):
    def __init__(self, channel, mean, std):
        self.channel = channel
        self.mean = np.array(mean, dtype='float32').reshape(1, 1, channel)
        self.std = np.array(std, dtype='float32').reshape(1, 1, channel)

    def normalize(self, x):
        return x * self.std + self.mean

    def invert_normalize(self, x):
        return (x - self.mean) / self.std


class ZeroToOneNormalization(Normalization):
    def normalize(self, x):
        return x * 255

    def invert_normalize(self, x):
        return x / 255


class NegativeOneToOneNormalization(Normalization):
    def normalize(self, x):
        return x * 255

    def invert_normalize(self, x):
        return x / 255


COLOR_BGR2BGR = 'COLOR_BGR2BGR'
COLOR_RGB2RGB = 'COLOR_RGB2RGB'
COLOR_RGB2BGR = 'COLOR_RGB2BGR'
COLOR_BGR2RGB = 'COLOR_BGR2RGB'


def tensor_to_numpy(img):
    return img.data.cpu().numpy().transpose(1, 2, 0)


def tensor_to_image(img, norm: Normalization, convert_color):
    img = norm.invert_normalize(tensor_to_numpy(img)).clip(0, 255).astype('uint8')
    if convert_color in [COLOR_RGB2BGR, COLOR_BGR2RGB]:
        new_image = np.zeros(img.shape, dtype='uint8')
        new_image[:, :, 0] = img[:, :, 2]
        new_image[:, :, 1] = img[:, :, 1]
        new_image[:, :, 2] = img[:, :, 0]
        img = new_image
    return img


def random_subset(dataset, size, seed=None):
    assert size <= len(dataset), 'subset size cannot larger than dataset'
    np.random.seed(seed)
    indexes = np.arange(len(dataset))
    np.random.shuffle(indexes)
    indexes = indexes[:size]
    return Subset(dataset, indexes)


def resize_image(img: torch.Tensor, height: int, width: int):
    return F.interpolate(img, size=(height, width), mode='bilinear', align_corners=False)


def downsampling(img: torch.Tensor, down_scale):
    H, W = img.size()[2:4]
    return F.interpolate(img, size=(int(H / down_scale), int(W / down_scale)), mode='bilinear', align_corners=False)


def upsampling(img: torch.Tensor, up_scale):
    H, W = img.size()[2:4]
    return F.interpolate(img, size=(int(H * up_scale), int(W * up_scale)), mode='bilinear', align_corners=False)


class MeanStdCalculator:
    def __init__(self, channel):
        self._channel = channel
        self._sqrt_mean = torch.zeros(channel, 1)
        self._mean = torch.zeros(channel, 1)
        self._n = 0

    def add(self, X):
        X = X.permute(1, 0, 2)
        channel, batch, length = X.size()
        total_length = batch * length
        X = X.reshape(channel, total_length)

        scale_1 = self._n / (self._n + total_length)
        scale_2 = total_length / (self._n + total_length)

        self._sqrt_mean = scale_1 * self._sqrt_mean + \
            scale_2 * X.pow(2).mean(dim=1).unsqueeze(1)
        self._mean = scale_1 * self._mean + \
            scale_2 * X.mean(dim=1).unsqueeze(1)

        self._n += total_length

    def mean(self):
        return self._mean.reshape(-1)

    def std(self):
        return torch.sqrt(self._sqrt_mean - self._mean.pow(2)).reshape(-1)
