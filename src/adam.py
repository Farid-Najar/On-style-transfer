import train
import matplotlib.pyplot as plt

if __name__ == '__main__':

    style_target = '../inputs/vangogh1.jpg'
    content_target = '../inputs/photo1.jpg'

    #Adam, content
    train.train(style_target, content_target, name_file = 'result_Adam', num_iters=6000, LBFGS = False, interval = 2000, training_path = 'result_Adam.jpg')
