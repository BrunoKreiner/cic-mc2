import torchio.transforms as transforms
import torchvision.transforms as pytransforms
import sys
import numpy as np
sys.path.insert(0,'../helpers/')
from helpers import miscellaneous as misc
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    Rand2DElastic,
    RandZoom,
    ScaleIntensity,
    EnsureType,
    NormalizeIntensity,
    RandGaussianSmooth,
    Resize,
    EnsureChannelFirst
)

def get_transformer(transformer_name):
    if transformer_name == 'None':
        return None, None
    elif transformer_name == 'Crop':
        return _crop_transformer(), _crop_transformer()
    elif transformer_name == 'Monai_Blur':
        return _monai_augment_blur_transformer(), None
    else:
        raise ValueError('Transformer is invalid or non-existent')

# for 2d images
def _crop_transformer():
    config = misc.get_config()
    return transforms.Compose(
        [
            NormalizeIntensity(),
            EnsureType(),
            pytransforms.Resize((config['IMAGE_RESIZE1'],config['IMAGE_RESIZE2']))
        ]
    )

def _monai_augment_blur_transformer():    
    config = misc.get_config()
    return transforms.Compose(
        [
            #Rand2DElastic(prob=0.3, spacing=(40,40), magnitude_range=(0,1), padding_mode='zeros'),
            RandGaussianSmooth(prob=0.5),
            RandRotate(prob=0.5),
            RandFlip(prob=0.4),
            #EnsureType()
        ]
    )