import train
import matplotlib.pyplot as plt

if __name__ == '__main__':
    style_target = '../inputs/uki2.jpg'
    content_target = '../inputs/photo1.jpg'

    #Uki
    train.train(style_target, content_target, name_file = 'result_uki', num_iters=600)
