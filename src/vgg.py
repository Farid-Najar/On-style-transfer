from collections import namedtuple
import torch
from torch import nn
from torchvision import models

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGG19(nn.Module):

    def __init__(self, requires_grad=False, show_progress=False):
        super(VGG19, self).__init__()

        self.layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4'
        ]

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        vgg_pretrained_features = models.vgg19(pretrained=True, progress=show_progress).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for i in range(4):
            self.slice1.add_module(self.layers[i], vgg_pretrained_features[i])

        for i in range(4, 9):
            self.slice2.add_module(self.layers[i], vgg_pretrained_features[i])

        for i in range(9, 14):
            self.slice3.add_module(self.layers[i], vgg_pretrained_features[i])

        for i in range(14, 23):
            self.slice4.add_module(self.layers[i], vgg_pretrained_features[i])

        for i in range(23, 30):
            self.slice5.add_module(self.layers[i], vgg_pretrained_features[i])
        del vgg_pretrained_features



    def forward(self, input_image):
        x = self.slice1(input_image)
        slice1 = x
        x = self.slice2(x)
        slice2 = x
        x = self.slice3(x)
        slice3 = x
        x = self.slice4(x)
        slice4 = x
        x = self.slice5(x)
        slice5 = x
        return (slice1, slice2, slice3, slice5), slice4

class VGG16(nn.Module):

    def __init__(self, requires_grad=False, show_progress=False):
        super(VGG16, self).__init__()

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        vgg_pretrained_features = models.vgg16(pretrained=True, progress=show_progress).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for i in range(4):
            self.slice1.add_module(str(i), vgg_pretrained_features[i])

        for i in range(4, 9):
            self.slice2.add_module(str(i), vgg_pretrained_features[i])

        for i in range(9, 16):
            self.slice3.add_module(str(i), vgg_pretrained_features[i])

        for i in range(16, 23):
            self.slice4.add_module(str(i), vgg_pretrained_features[i])


    def forward(self, input_image):
        x = self.slice1(input_image)
        slice1 = x
        x = self.slice2(x)
        slice2 = x
        x = self.slice3(x)
        slice3 = x
        x = self.slice4(x)
        return (slice1, slice2, x), slice3
