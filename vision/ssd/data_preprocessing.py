from ..transforms.transforms import *


class DivideByStd:
    def __init__(self, std):
        self.std = std

    def __call__(self, img, boxes=None, labels=None):
        return img / self.std, boxes, labels


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop_v2(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            DivideByStd(std),  # ← replaced lambda
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            DivideByStd(std),  # ← replaced lambda
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            DivideByStd(std),  # ← replaced lambda
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image