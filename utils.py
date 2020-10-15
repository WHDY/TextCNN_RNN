import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# make file
def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


# keep a record to training procedure
def record_train(writer, L, loss, train_acc, acc, T, t):
    writer.add_scalar('training loss', loss, t)
    writer.add_scalar('training accuracy', acc, t)
    L.append(loss)
    train_acc.append(acc)
    T.append(t)


def record_test(writer, test_acc, acc, T, t):
    writer.add_scalar('test accuracy', acc, t)
    test_acc.append(acc)
    T.append(t)


def make_curve(train_loss, train_acc, test_acc, train_t, test_t, save_path):
    plt.title('train accuracy')
    plt.xlabel('iters')
    plt.ylabel('accuracy')
    plt.plot(train_t, train_acc, color='green')
    plt.savefig(os.path.join(save_path, 'train_acc.png'))

    plt.cla()
    plt.title('train loss')
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.plot(train_t, train_loss, color='red')
    plt.savefig(os.path.join(save_path, 'train_loss.png'))

    plt.cla()
    plt.title('test accuracy')
    plt.xlabel('iters')
    plt.ylabel('accuracy')
    plt.plot(test_t, test_acc, color='green')
    plt.savefig(os.path.join(save_path, 'test_acc.png'))


