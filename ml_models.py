import torch
import torch.nn as nn
import torchio as tio
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch.nn.functional as F
import monai
from monai.networks.nets import DenseNet121, AHNet, ResNet
from monai.networks.nets.resnet import resnet10, resnet18, resnet50
import torchxrayvision as xrv


def get_model(model_name=None):
    if model_name == 'TestCNN':
        return TestCNN(CONFIG)
    elif model_name == 'EncoderCNN':
        return EncoderCNN()
    elif model_name == 'CognitiveTestMLP':
        return CognitiveTestMLP()
    elif model_name == 'FusionModel':
        return FusionModel()
    elif model_name == 'LeNet5':
        return LeNet5()
    elif model_name == 'BaselineNet':
        return BaselineNet()
    elif model_name == 'DenseNet121':
        return DenseNet121(spatial_dims=2, in_channels=1, out_channels=3)
    elif model_name == 'DenseNet121_XRV':
        return nn.Sequential(xrv.models.DenseNet(weights="densenet121-res224-all"),
                             nn.Linear(18,3))
    elif model_name == 'AHNet':
        return AHNet(spatial_dims=2, n_input_channels =1, num_classes =3, pretrained=CONFIG['PRETRAINED'])
    elif model_name == 'ResNet10':
        return resnet10(spatial_dims=2, n_input_channels =1, num_classes =3)
    elif model_name == 'ResNet18':
        return resnet18(spatial_dims=2, n_input_channels =1, num_classes =3)
    elif model_name == 'ResNet50':
        return resnet50(spatial_dims=2, n_input_channels =1, num_classes =3)
    
    else:
        print('[WARN] Unknown model selected')
        

# TODO: Add more parameters to Models so that it's configurable
class EncoderCNN(nn.Module):
    """
    Encodes a 2D image into output neurons
    Attributes
    ----------
    resnet: use resnet model structure
    linear: linear layer after resnet model
    bn: batchnormalisation layer definition
    ----------
    Methods
    ----------
    forward(images): takes images and does forward feed calculation
    """
    def __init__(self, CNN_out_size, train_CNN=False):
        """
        Parameters
        ----------
        CNN_out_size: int | Amount of ouput neurons
        train_CNN: boolean | use pretrained model yes/no
        """
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        
        #change to 3D conv net with nn.Conv3d()
        #resnet = models.resnet18(pretrained=True)
        #modules = list(resnet.children())[:-1]
        #self.resnet = nn.Sequential(*modules)
        
        self.convolute = nn.Conv3d(1, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        
        #self.linear = nn.Linear(resnet.fc.in_features, CNN_out_size)
        #replace hardcoded size to formula: 
        
        convolution_size = get_conv_output_shape(50, kernel_size=(3, 5, 2), stride=(2, 1, 1), pad=(4, 2, 0), dilation = 1, type_of_convolution = "3D")
        self.linear = nn.Linear(33 * np.prod(convolution_size), CNN_out_size)
        
        self.bn1 = nn.InstanceNorm3d(33 * np.prod(convolution_size))
        self.bn2 = nn.BatchNorm1d(CNN_out_size, momentum=0.01)
        
    def forward(self, images):
        """
        Parameters
        ----------
        images: torch.Tensor(batch size, _, _, _,) |  
        """
        
        #with torch.no_grad():
        #features = self.resnet(images)
        
        features = nn.functional.relu(self.convolute(images))
        features = self.bn1(features)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn2(features)
        features = torch.nn.functional.softmax(features, dim=1)
        return features
    
class CognitiveTestMLP(nn.Module):
    """
    Simple MLP for tabular data
    Attributes
    ----------
    layers: nn.Sequential object with MLP model setup
    ----------
    Methods
    ----------
    forward(features): takes label images and does forward feed calculation
    """
    def __init__(self, MLP_out, MLP_in):
        """
        Parameters
        ----------
        MLP_out: int | Amount of ouput neurons
        MLP_in: int | Amount of labels
        """
        super(CognitiveTestMLP, self).__init__()
        
        self.layers = nn.Sequential(
          nn.Linear(MLP_in, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, MLP_out)
        )
    
    def forward(self, features):
        return self.layers(features)

class FusionModel(nn.Module):
    """
    Fuses the CNN for images and MLP for labels together using a simple concatenate function
    Attributes
    ----------
    encoderCNN: EncoderCNN object
    cognitiveTestMLP: CognitiveTestMLP
    fusion: fusion layer
    fc: a final linear layer
    ----------
    Methods
    ----------
    forward(features): takes label images and does forward feed calculation
    """
    def __init__(self, CNN_out_size, MLP_out, MLP_in, fusion_output_size, num_classes):
        """
        Parameters
        ----------
        CNN_out_size: int | Amount of ouput neurons of CNN
        MLP_out: int | Amount of ouput neurons of MLP
        MLP_in: int | Amount of labels
        fusion_output_size: int | Amount of output neurons of fusion/concatenation layer
        num_classes: int | Amount of output classes/labels
        """
        super(FusionModel, self).__init__()
        self.encoderCNN = EncoderCNN(CNN_out_size)
        self.cognitiveTestMLP = CognitiveTestMLP(MLP_out, MLP_in)
        
        self.fusion = torch.nn.Linear(
            in_features=(MLP_out + CNN_out_size), 
            out_features=fusion_output_size
        )
        
        self.fc = torch.nn.Linear(
            in_features=fusion_output_size, 
            out_features=num_classes
        )
        
    def forward(self, images, test_data):
        """
        Data goes through the 2 early models then it gets put together and pushed through a linear layer and another final linear layer 
        Parameters
        ----------
        images: int | image as torch.Tensor
        test_data: int | tabular data
        """
        CNN_features = self.encoderCNN(images)
        MLP_features = self.cognitiveTestMLP(test_data)
        
        #here play around with squeeze() and unsqueeze() when using 3D images
        combined = torch.cat(
            [MLP_features.squeeze(1), CNN_features], dim=1
        )
        
        #here opportunity for another BN layer or 
        
        fused = torch.nn.functional.relu(self.fusion(combined))
        logits = self.fc(fused)
        pred = torch.nn.functional.softmax(logits)
        
        return pred

    
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        CONFIG = misc.get_config()

        self.cnns = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten())

        n_channels = self.cnns(torch.empty(CONFIG['BATCH_SIZE'], 1, CONFIG['IMAGE_RESIZE1'], CONFIG['IMAGE_RESIZE2'])).size(-1)

        print(n_channels)
        self.dense = nn.Sequential(
            nn.Linear(n_channels, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 3))

    def forward(self, x):
        features = self.cnns(x)
        out = self.dense(features)
        return out


class BaselineNet(nn.Module):
    """
    Baseline CNN-Model for binary classification. Takes grayscale images with shape (254,254) as input
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.pool1 = nn.MaxPool2d()
        self.pool2 = nn.MaxPool2d()

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()


    def forward(self, x):
        x = self.pool1 =(F.relu(self.conv1(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class TestCNN(nn.Module):
    
    def __init__(self, CONFIG):
        super().__init__()
        
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.4),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        """self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(65536, 3),
            nn.Softmax(dim=1)
        )"""
        
        self.cnn_to_mlp = nn.Sequential(
          nn.Flatten(),
          nn.Linear(16384, 512),
          nn.ReLU(),
          nn.Linear(512, 128),
          nn.ReLU(),
          nn.Linear(128, 3),  
          nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        
        print("x_shape: ", x.shape)
        
        model = nn.Sequential(
            self.cnn_layers,
            self.cnn_to_mlp
        )
        
        return model(x)
    
        