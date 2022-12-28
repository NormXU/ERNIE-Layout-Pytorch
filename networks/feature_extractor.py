import torch
from torchvision import transforms
import torchvision.transforms.functional as F

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


class ErnieFeatureExtractor:

    def __init__(self, imagenet_default_mean_and_std=False, size=224, interpolation='bilinear', **kwargs):
        self.size = size
        self.interpolation = self._pil_interp(interpolation)
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.patch_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))])

    def _pil_interp(self, method):
        if method == 'bicubic':
            return F.InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return F.InterpolationMode.LANCZOS
        elif method == 'hamming':
            return F.InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return F.InterpolationMode.BILINEAR

    def __call__(self, img, **kwargs):
        for_patches = F.resize(img, [self.size, self.size], self.interpolation)
        return self.patch_transform(for_patches)