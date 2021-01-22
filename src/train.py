import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import vgg
#import os
import numpy as np
from PIL import Image

DATA_PATH = '~/On-style-transfer/data/datasets/'
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]
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

    training_img_content_feature = training_img_features[-1].squeeze(axis=0)
    content_loss = nn.MSELoss(reduction='mean')(content_feature, training_img_content_feature)/2

    training_img_style_features = [gram(training_img_features[0][i]) for i in range(len(style_features))]
    style_loss = 0.
    for i in range(len(style_features)):
        #N, M = style_features[i][0].shape
        #print(N, M)
        style_loss += w[i]*nn.MSELoss(reduction='sum')(style_features[i][0], training_img_style_features[i][0])#/((N**2)*(M**2))
    style_loss /= len(style_features)

    total = content_weight*content_loss + style_weight*style_loss

    return total


def load_img(path):
    image = np.array(Image.open(path)).astype(np.float32)
    image /= 255.0
    transform = transforms.Compose([
        #transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255,
                             std=[1,1,1]),])
    image = transform(image).to(DEVICE).unsqueeze(0)
    return image

def save_img(img,path):
    # convert torch tensor to PIL image and then save to path
    img1 = img.squeeze(axis=0).to('cpu').detach().numpy()
    img1 = np.moveaxis(img1, 0, 2)
    img2 = np.copy(img1)
    img2 += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    image = np.clip(img2, 0, 255).astype('uint8')
    image=transforms.ToPILImage()(image) # denormalize tensor before convert
    image.save(path)

def train(style_target, content_target, name_file = 'result', ratio = 10e-3, num_iters = 1000, LBFGS = True, random_initialization = False, interval = 100, training_path = None):
    style_img = load_img(style_target)
    content_img = load_img(content_target)
    #learning parameters
    learning_rate = 1e-2
    #the values are arbitrary. Most important is the ratio between them
    style_weight = 1
    content_weight = ratio
    save_path=f'../outputs/{name_file}.jpg'

    #we make a random image to train or we take the content image
    if training_path == None :
        if random_initialization :
            train_img = Variable(torch.randn(content_img.size(), device = DEVICE),requires_grad = True)
        else :
            train_img = Variable(load_img(content_target), requires_grad = True)
    else :
        train_img = Variable(load_img(training_path), requires_grad = True)
    #tracers
    loss_list = []
    min_loss = float("inf")
    best_img = train_img

    #we crerate the L-BFGS/Adam optimizer as suggested in the paper
    if LBFGS :
        optimizer = optim.LBFGS((train_img,), lr = learning_rate, line_search_fn='strong_wolfe')#, history_size = 20)
    else :
        optimizer = optim.Adam((train_img,), lr = learning_rate)
    net = vgg.VGG19().to(DEVICE)

    #We extract features from style image and content image
    forwarded_style_img, _ = net(style_img)
    style_img_features = [gram(forwarded_style_img[i]) for i in range(len(forwarded_style_img))]
    content_img_features = net(content_img)[-1].squeeze(axis=0)
    #del style_img, content_img

    for counter in range(num_iters):
            def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    l = loss(net, train_img, style_features=style_img_features, content_feature=content_img_features,
                         content_weight=content_weight, style_weight=style_weight, w=[i for i in range(len(style_img_features))])
                    loss_list.append(l.detach().numpy())
                    if l.requires_grad:
                        l.backward(retain_graph=True)
                    return l
            try:
                optimizer.step(closure)
                print(f'loss at {counter} = {loss_list[-1]}')
                if min_loss > loss_list[-1]:
                    min_loss = loss_list[-1]
                    best_img = train_img
                if counter%interval == 0:
                    save_img(best_img, f'../outputs/{name_file+str(counter)}.jpg')
                
            except :
                print('###########################################################')
                print(f'we did {counter} iterations.')
                print('###########################################################')
                break
                

    save_img(best_img, save_path)
    return loss_list
