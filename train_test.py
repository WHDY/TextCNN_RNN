import os
import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn import metrics

from dataGenerator import loadIMDBData, dataLoader
from Models import CNNTextClassifier, LSTMTextClassifier

from utils import test_mkdir, record_train, record_test, make_curve

# ------------------------------------------- 训练需要使用的参数 ---------------------------------------------------- #
parser = argparse.ArgumentParser(description='Ch/En Text Classification')

# 是否使用GPU
parser.add_argument('--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')

# 模型参数
parser.add_argument('--model', type=str, default='CNN', help='choose a model: CNN, LSTM')
parser.add_argument('--num-filter', type=int, default=256, help='number of filters when using CNN')
parser.add_argument('--filter-shape', type=str, default='2,3,4', help='shape of filter when using CNN, every number must split by "," ')
parser.add_argument('--classes', type=int, default=2, help='number of classes')
parser.add_argument('--num-layers', type=int, default=2, help='number of layers when using LSTM')
parser.add_argument('--hl-size', type=int, default=128, help='size of hidden layer when using LSTM')

# 数据参数
parser.add_argument('--embed-path', type=str, default='./pretrained_WordVectors/glove.6B.200d.gen.npy', help='path of pretrained word-vectors')
parser.add_argument('--voca-len', type=int, default=20000, help='length of vocabulary')
parser.add_argument('--text-len', type=int, default=200, help='fixed length of every text')

# 训练参数
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dp', type=float, default=0.5, help='dropout rate')
parser.add_argument('--batch-size', type=int, default=100, help='batch size')
parser.add_argument('--epochs', type=int, default=15, help='epochs')
parser.add_argument('--record-freq', type=int, default=15, help='frequency of recording training procedure')
parser.add_argument('--save-freq', type=int, default=5, help='save frequency(eopch)')

# 结果相关
parser.add_argument('--ckpt-path', type=str, default='./checkPoint', help='path to save checkpoint file')
parser.add_argument('--out-file', type=str, default='./logdir', help="file path to store the results")
parser.add_argument('--confusion', type=bool, default=True, help='whether to calculate confution matrix')


def calculate_confusion_matrix(Net, testDataLoader, length, device):
    Net.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for k in range(length // 100):
        batchTexts, batchLabels = testDataLoader.next_batch()
        batchTexts = torch.tensor(batchTexts).to(device)

        test_preds = Net(batchTexts)
        test_preds = torch.argmax(test_preds, dim=1)

        if device == torch.device('cuda'):
            test_preds = test_preds.to(torch.device('cpu'))

        predict_all = np.append(predict_all, test_preds.numpy())
        labels_all = np.append(labels_all, batchLabels)

    confusion = metrics.confusion_matrix(labels_all, predict_all)
    return confusion


def train_and_test():
    test_mkdir(args['ckpt_path'])  # make file
    test_mkdir(args['out_file'])
    writer = SummaryWriter(args['out_file'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']  # 是否使用GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load data and construct data loader
    trainTextsIds, train_labels, testTextsIds, test_labels, embedding = loadIMDBData(args['text_len'], args['voca_len'], args['embed_path'])
    trainDataLoader = dataLoader(trainTextsIds, train_labels, args['batch_size'], shuffle=True)
    testDataLoader = dataLoader(testTextsIds, test_labels, args['batch_size'], shuffle=False)

    # construct model
    filter_shape = [int(d) for d in args['filter_shape'].strip().split(',')]
    Net = None
    if args['model'] == 'CNN':
        Net = CNNTextClassifier(torch.tensor(embedding), args['num_filter'], filter_shape, args['dp'], args['classes'])
    elif args['model'] == 'LSTM':
        Net = LSTMTextClassifier(torch.tensor(embedding), args['num_layers'], args['hl_size'], args['dp'], args['classes'])
    else:
        pass
    Net.to(device)

    # set optimizer
    optimizer = optim.Adam(Net.parameters(), lr=args['lr'])

    # train
    train_acc = []
    train_loss = []
    train_t = []

    test_acc = []
    test_t = []
    acc = 0.0
    batchsize = args['batch_size']
    B = trainTextsIds.shape[0] // batchsize
    for epoch in tqdm(range(args['epochs'])):
        print('epoch: {}'.format(epoch + 1))
        for i in range(B):
            Net.train()
            batchTexts, batchLabels= trainDataLoader.next_batch()
            batchTexts = torch.tensor(batchTexts).to(device)
            batchLabels = torch.tensor(batchLabels).to(device)

            preds = Net(batchTexts)
            loss = F.cross_entropy(preds, batchLabels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch*B + i + 1) % args['record_freq'] == 0:
                with torch.no_grad():
                    Net.train()
                    train_preds = torch.argmax(preds, dim=1)
                    acc = (train_preds == batchLabels).float().mean().item()
                    record_train(writer, train_loss, loss.item(), train_acc, acc, train_t, epoch * B + i + 1)

                    Net.eval()
                    acc_sum = 0.0
                    num = 0
                    for k in range(testTextsIds.shape[0] // batchsize):
                        batchTexts, batchLabels = testDataLoader.next_batch()
                        batchTexts = torch.tensor(batchTexts).to(device)
                        batchLabels = torch.tensor(batchLabels).to(device)

                        test_preds = Net(batchTexts)
                        test_preds = torch.argmax(test_preds, dim=1)
                        acc_sum += (test_preds==batchLabels).float().mean().item()
                        num += 1
                record_test(writer, test_acc, acc_sum/num, test_t, epoch*B + i + 1)
                print('testing accuracy is: {}%'.format(acc_sum*100/num))

        if (epoch + 1) % args['save_freq'] == 0:
            torch.save(Net.state_dict(),
                       os.path.join(args['ckpt_path'], '{}_E{}.ckpt'.format(args['model'], epoch + 1)))

    # make the training curve
    make_curve(train_loss, train_acc, test_acc, train_t, test_t, args['out_file'])

    # calculate confusion matrix
    if args['confusion']:
        confusion_matrix = calculate_confusion_matrix(Net, testDataLoader, testTextsIds.shape[0], device)
        print('confusion_matrix is {}'.format(confusion_matrix))


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    train_and_test()

