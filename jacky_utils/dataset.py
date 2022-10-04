import numpy as np

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