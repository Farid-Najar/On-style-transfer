import train
import matplotlib.pyplot as plt

if __name__ == '__main__':
    style_target = '../inputs/vangogh1.jpg'
    content_target = '../inputs/photo1.jpg'

    train.train(style_target, content_target, name_file = 'result_style', num_iters=1000, LBFGS = True, training_path = style_target)
