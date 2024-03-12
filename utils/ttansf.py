import torch.nn.functional as F
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class SquarePad(object):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(  # 宽高比调整成1:1，多余的部分填充黑色，图像居中
                1.0,
                position="center-center").to_deterministic()
        ])
    def __call__(self, img):
        img = self.augmentations(image=img)
        return img

class ToTensor(object):
    def __init__(self, ):
        pass
    def __call__(self, img):
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)
        return img

DEFAULT_TRANSFORMS = transforms.Compose([
    SquarePad(),
    ToTensor()
])
# img_path = "../data/custom/train.txt"

# dataset1 = ListDataset(
#     img_path,
#     img_size=416,
#     multiscale=False,
#     transform=DEFAULT_TRANSFORMS)

# print(dataset1[0][0],dataset1[0][1].shape,dataset1[0][2].shape)
