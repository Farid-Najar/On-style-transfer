from typing import Any

import numpy as np
import scipy.io
import torch
from torch import nn

VGG_FILE = 'imagenet-vgg-verydeep-19.mat'
MEAN_PIXEL = [123.68, 116.779, 103.939] #obtained from np.mean(mean, axis=(0, 1)) see functions bellow

def net_relu(path, input_image):
    """
    Gives the VGG network using relu as activation function
    @param path: the path to data
    @param input_image: the input image
    """

    #we define our layers as they are defined in the paper
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    data = scipy.io.loadmat(path)   #we load the vgg net
    mean = data['normalization'][0][0][0]   #we take the mean of the data
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]     #we take the weights

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        type = name[:4]     #as you can see in layers, the type of the layer is described by the first four letters
        if type == 'conv':
            print("")
        elif type == "relu":
            current = nn.relu(current)
        elif type == "pool":

        net[name] = current
    return net



"""
def net_tanh(path, input_image):
    
    #Gives the VGG network using tanh as activation function
    #@param path: the path to data
    #@param input_image: the input image
    
    layers = (
        'conv1_1', 'tanh1_1', 'conv1_2', 'tanh1_2', 'pool1',

        'conv2_1', 'tanh2_1', 'conv2_2', 'tanh2_2', 'pool2',

        'conv3_1', 'tanh3_1', 'conv3_2', 'tanh3_2', 'conv3_3',
        'tanh3_3', 'conv3_4', 'tanh3_4', 'pool3',

        'conv4_1', 'tanh4_1', 'conv4_2', 'tanh4_2', 'conv4_3',
        'tanh4_3', 'conv4_4', 'tanh4_4', 'pool4',

        'conv5_1', 'tanh5_1', 'conv5_2', 'tanh5_2', 'conv5_3',
        'tanh5_3', 'conv5_4', 'tanh5_4'
    )
    return 0
"""