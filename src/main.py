import train
import matplotlib.pyplot as plt

if __name__ == '__main__':
    style_target = '../inputs/vangogh1.jpg'
    content_target = '../inputs/photo1.jpg'
    """
    list_loss_content = train.train(style_target, content_target, LBFGS=False)
    plt.plot(range(len(list_loss_content)), list_loss_content)
    plt.savefig('loss_content.png')
    """
    #random
    list_loss_random = train.train(style_target, content_target, name_file = 'result_random', num_iters=1000, LBFGS = True, random_initialization = True)
    plt.plot(range(len(list_loss_random)), list_loss_random)
    plt.savefig('loss_random.png')


