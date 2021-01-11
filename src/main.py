import train
import matplotlib.pyplot as plt

if __name__ == '__main__':
    style_target = '/home/farid/Master/M1/S1/Deep_Learning/Project/On-style-transfer/inputs/vangogh1.jpg'
    content_target = '/home/farid/Master/M1/S1/Deep_Learning/Project/On-style-transfer/inputs/photo1.jpg'
    list_loss = train.train(style_target, content_target)
    plt.plot(range(len(list_loss)), list_loss)
    plt.show()
    #parser
    print("begin")
