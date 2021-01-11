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
    #print('###########################################################')
    #print(f"gram fun : x.size() = {x.size()}")
    #print('###########################################################')
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    res = features.bmm(features_t)
    if normalize:
        res /= ch * h * w
    return res

def loss(net, training_img, style_features, content_feature, content_weight, style_weight, w):
    training_img_features = net(training_img)
    #print('###########################################################')
    #print(f"loss fun : features size = {len(training_img_features[0].size())}")
    #print('###########################################################')
    training_img_style_features = [gram(training_img_features[i]) for i in range(len(style_features))]
    style_loss = 0.

    for i in range(len(style_features)):
        style_loss += w[i]*nn.MSELoss(reduction='mean')(style_features[i][0], training_img_style_features[i][0])
    style_loss /= len(style_features)

    training_img_content_feature = training_img_features[-1].squeeze(axis=0)
    content_loss = nn.MSELoss(reduction='sum')(content_feature, training_img_content_feature)/2

    return content_weight*content_loss + style_weight*style_loss


def load_img(path):
    image = Image.open(path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = IMAGENET_MEAN_255, std = (1, 1, 1)),])
    with torch.no_grad():
        image = Variable(transform(image).to(DEVICE))
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

def train(style_target, content_target):
    style_img = load_img(style_target)
    content_img = load_img(content_target)
    #learning parameters
    learning_rate = 1e-2
    #the values are arbitrary. Most important is the proportion between them
    style_weight = 1
    content_weight = 1e-3
    #number of iterations or epoch
    num_iters = 2000
    save_path='../outputs/result.jpg'

    #we make a random image to train
    train_img = Variable(torch.randn(content_img.size(), device = DEVICE),requires_grad = True)

    #tracers
    loss_list = []
    min_loss = float("inf")
    best_img = train_img

    #we crerate the L-BFGS optimizer as suggested in the paper
    optimizer = optim.Adam([train_img], lr = learning_rate)#optim.LBFGS([train_img], lr = learning_rate, history_size=20)
    net = vgg.VGG19().to(DEVICE)

    #We extract features from style image and content image
    *forwarded_style_img, _ = net(style_img)
    style_img_features = [gram(forwarded_style_img[i]) for i in range(len(forwarded_style_img))]
    content_img_features = net(content_img)[-1].squeeze(axis=0)
    del style_img, content_img

    if torch.is_grad_enabled() :
        optimizer.zero_grad()
    for _ in range(num_iters):
        def closure ():
            l = loss(net, train_img, style_features=style_img_features, content_feature=content_img_features,
                 content_weight=content_weight, style_weight=style_weight, w=[1 for _ in range(len(style_img_features))])
            loss_list.append(l)
            l.backward(retain_graph=True)
            optimizer.zero_grad()
            return l
        try:
            optimizer.step(closure)

            if min_loss > loss_list[-1]:
                min_loss = loss_list[-1]
                best_img = train_img
        except O:
            break

    save_img(best_img, save_path)
    return loss_list
