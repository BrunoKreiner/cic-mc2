import torchio as tio
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image

import pandas as pd
import os
from sklearn.utils import shuffle

class MRIDataset(Dataset):
    """
    Implements __getitem__ method to get items for batches
    Attributes
    ----------
    image_path (string): Path to the image folder
    test_path (string: Path to where tabular data is stored
    annotation_path (string): path to annotation where label is stored
    transform (object, callable): Optional transform to be applied
        on a sample.
    labels: labels extracted from annotation_path
    images: iterable list of paths to all images (could also be images themselves)
    test_results: test results extracted from test_path
    ----------
    Methods
    ----------
    forward(images): takes images and does forward feed calculation
    """
    def __init__(self, df, shuffle_ = True, transform=None):
        """
        Parameters:
            image_path (string): Path to the image folder
            test_path (string: Path to where tabular data is stored
            annotation_path (string): path to annotation where label is stored
            transform (object, callable): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        # fixme should be handled in another way. maybe in configuration file?
        self.idx_to_label = {
            'CN': 0,
            'MCI': 1,
            'AD': 2
        }

        self.df = df
        
        if shuffle_ == True:
            self.df = shuffle(self.df)

    def __len__(self):
        """
        returns amount of images in total
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Parameters:
            idx (int): index to image
        """
        img = tio.ScalarImage(self.df.iloc[idx].filename)
        img = img.data
        #get image and caption by id
        label = self.df.iloc[idx].Group
        label = self.idx_to_label[label]
        label = nn.functional.one_hot(torch.tensor(label), num_classes = 3).to(torch.float32)
                
        if self.transform is not None:
            try:
                img = self.transform(img)
            except FloatingPointError:
                print(f"img {self.df.iloc[idx].filename} couldn't be transformed, floating point error: overflow encountered in float_scalars")

        img = img[0]
        return {"images": img, "labels": label } #, test_results