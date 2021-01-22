import train
import matplotlib.pyplot as plt

if __name__ == '__main__':
    style_target = '../inputs/snow.jpg'
    content_target = '../inputs/photo1.jpg'

    train.train(style_target, content_target, name_file = 'result_snow', ratio=10e-1, num_iters=1000, LBFGS = True)
