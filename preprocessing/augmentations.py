import albumentations as albu
from functools import partial


def downsample(x, factor=2, **kwargs):
    return x[::factor, ::factor, ...]


def get_training_augmentation(height, width, downsample_factor=1):
    downsample_fn = partial(downsample, factor=downsample_factor)
    train_transform = [
        albu.Lambda(image=downsample_fn, mask=downsample_fn),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
        albu.RandomCrop(height=height, width=width, always_apply=True),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(height, width, downsample_factor=1):
    """Add paddings to make image shape divisible by 32"""
    downsample_fn = partial(downsample, factor=downsample_factor)
    test_transform = [
        albu.Lambda(image=downsample_fn, mask=downsample_fn),
        albu.PadIfNeeded(min_height=height, min_width=width),
        albu.CenterCrop(height=height, width=width)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn, downsample_factor=1):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    downsample_fn = partial(downsample, factor=downsample_factor)
    _transform = [
        albu.Lambda(image=downsample_fn, mask=downsample_fn),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
