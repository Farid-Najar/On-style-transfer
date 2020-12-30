from collections import namedtuple
import torch
from torchvision import models

#VGG_FILE = 'imagenet-vgg-verydeep-19.mat'
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]

class VGG19(nn.Module):

    def __init__(self, requires_grad=False, show_progress=False):
        super(VGG19, self).__init__()
        #we define our layers as they are defined in the paper
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

        self.slices_name = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']
        vgg_pretrained_features = models.vgg19(pretrained=True, progress=show_progress).features

        # as specified in the paper, we use the fourth layer to represent content
        self.content_feature_maps_index = layers.index('relu4_4')
        # all layers are used for style representation except relu4_4
        self.style_feature_maps_indices = [layers.index(self.slices_name[i]) for i in range(len(self.slices_name))]
        self.style_feature_maps_indices.remove(self.layers.index('relu4_4'))  # relu4_4
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        #from 'conv1_1' to 'relu1_2'
        for i in range(4):
            self.slice1.add_module(self.layers[i], vgg_pretrained_features[i])
        #from 'pool1' to 'relu2_2'
        for i in range(4, 9):
            self.slice2.add_module(self.layers[i], vgg_pretrained_features[i])
        #from 'pool2' to 'relu3_4'
        for i in range(9, 18):
            self.slice3.add_module(self.layers[i], vgg_pretrained_features[i])
        #from 'pool3' to 'relu4_4'
        for i in range(18, 27):
            self.slice4.add_module(self.layers[i], vgg_pretrained_features[i])
        #from 'pool4' to 'relu5_4'
        for i in range(27, 36):
            self.slice5.add_module(self.layers[i], vgg_pretrained_features[i])



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
        #vgg_outputs = namedtuple("VGG19Outputs", self.slices_name)
        #output = vgg_outputs(slice1, slice2, slice3, slice4, slice5)
        return (slice1, slice2, slice3, slice5, slice4)#output

"""  
def net_relu(path, input_image):
    Gives the VGG network using relu as activation function
    @param path: the path to data
    @param input_image: the input image



    data = scipy.io.loadmat(path)   #we load the vgg net
    mean = data['normalization'][0][0][0]   #we take the mean of the data
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]     #we take the weights

    net = {}
    current = input_image
    for i, name in enumerate(layers):
        type = name[:4]     #as you can see in layers, the type of the layer is described by the first four letters
        if type == 'conv':

            cnn_layer = nn.Conv2d()
        elif type == "relu":
            current = nn.relu(current)
        elif type == "pool":

        net[name] = current
    return net




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
