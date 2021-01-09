import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import vgg
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

DATA_PATH = '~/On-style-transfer/data/datasets/'
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gram(x, normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    res = features.bmm(features_t)
    if normalize:
        res /= ch * h * w
    return res

def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

def load_img(path):
    # a function to load image and transfer to Pytorch Variable.
    image = Image.open(path)
    transform = transforms.Compose([
        transforms.ToTensor(),#Converts (H x W x C) of[0, 255] to (C x H x W) of range [0.0, 1.0].
        transforms.Normalize(mean = IMAGENET_MEAN_255, std = (1, 1, 1)),])
    image = Variable(transform(image).to(DEVICE), volatile=True)
    image = image.unsqueeze(0)
    return image

def show_img(img):
    # convert torch tensor to PIL image and then show image inline.
    img=transforms.ToPILImage()(img[0].data*0.5+0.5) # denormalize tensor before convert
    plt.imshow(img,aspect=None)
    plt.axis('off')
    plt.gcf().set_size_inches(8, 8)
    plt.show()

def save_img(img,path):
    # convert torch tensor to PIL image and then save to path
    img=transforms.ToPILImage()(img[0].data*0.5+0.5) # denormalize tensor before convert
    img.save(path)

def train(artist_name, style_target):
    content_path = DATA_PATH+f'{artist_name.lower()}2photo/trainB'
    style_image = load_img(DATA_PATH+f'{artist_name.lower()}2photo/trainA', style_target)
    #learning parameters
    learning_rate = 1e-2
    #the values are arbitrary. Most important is the proportion between them
    style_weight = 1
    content_weight = 1e-3
    #number of iterations or epoch
    num_iters = 500
    save_path='saver/fns.ckpt'

    #tracers
    loss_list = []
    min_loss = float("inf")

    #we make a random image to train
    train_img = Variable(torch.randn(content_img.size()),requires_grad = True)
    #we crerate the L-BFGS optimizer as suggested in the paper
    optimizer = optim.LBFGS([train_img], lr = learning_rate, history_size=50)
    L2loss = nn.MSELoss(size_average=False)
    net = vgg.VGG19()

    for _ in range(num_iters):
        optimizer.zero_grad()

        *style_features, content_features = net.forward(train_img)
        # compute total loss
        style_loss = [L2loss(style_features[i],style_refs[i]) for i in range(len(style_features))]
        #a small trick to balance the influnce of diffirent style layer
        mean_loss = sum(style_loss).data[0]/len(style_features)

        style_loss = sum([(mean_loss/l.data[0])*l*STYLE_LAYER_WEIGHTS[i]
                          for i,l in enumerate(style_loss)])/len(style_features)

        content_loss = sum([l2loss(content_features[i],content_refs[i])
                            for i in range(len(content_refs))])/len(content_refs)
        total_loss = style_weight*style_loss+content_weight*content_loss
        total_loss.backward()

        if min_loss > loss_history[-1]:
            min_Loss = loss_history[-1]
            best_img = train_img

        optimizer.step()