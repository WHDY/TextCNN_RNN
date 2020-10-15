import os
from tqdm import tqdm
from collections import Counter

import nltk
from nltk.tokenize import RegexpTokenizer

import numpy as np


UNK = '<UNK>'  # 未知词替代符号
PAD = '<PAD>'  # padding符号

def loadIMDBData(maxSenLen, vocabLen, embeddingPath=None):
    '''
    load data from IMDB dataset and glove word_vectors
    :param maxSenLen: 固定文本长度
    :param vocabLen:  词汇量
    :return: numpy数组类型的训练数据，训练标签，测试数据，测试标签，词向量矩阵
    '''

    # ------------------------------------- 读取文本数据及标签 --------------------------------------- #
    print('start reading texts and labels...')
    imdb_dir = r'./data/aclImdb'  # path
    train_dir = os.path.join(imdb_dir, 'train')
    test_dir = os.path.join(imdb_dir, 'test')

    # read data
    def read_data(path):
        labels = []
        texts = []
        for label_type in ['pos', 'neg']:
            dir_name = os.path.join(train_dir, label_type)
            for fname in os.listdir(dir_name):
                if fname[-4:] == '.txt':
                    f = open(os.path.join(dir_name, fname), encoding='utf-8')
                    texts.append(f.read().lower())  # 注意！转化为小写！
                    f.close()
                if label_type == 'pos':
                    labels.append(1)
                else:
                    labels.append(0)
        return texts, labels

    train_texts, train_labels = read_data(train_dir)  # 用于训练的文本数据
    test_texts, test_labels = read_data(test_dir)   # 用于测试的文本数据

    # --------------------------------------- 创建词汇列表 ---------------------------------------- #
    print('start constructing vocabulary list...')
    sent_tokenize = nltk.sent_tokenize  # 按句子进行分割
    word_tokenize = RegexpTokenizer(r'\w+').tokenize  # 按词进行分割

    def flatten(l):  # 变为一维的单词List
        return [item for sublist in l for item in sublist]

    def get_paragraph_words(text):  # 分词
        return (flatten([word_tokenize(s) for s in sent_tokenize(text)]))

    # 取出现频率最高的前vocabLen的单词作为词汇表
    vocab_counter = Counter(flatten([get_paragraph_words(text) for text in train_texts]))  # 词频字典
    vocab_list = sorted([_ for _ in vocab_counter.items()], key=lambda x: x[1], reverse=True)[:vocabLen]

    # print(len(vocab_counter))
    # 建立词汇字典： word -> idx
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})  # UNK: 用于所有词汇表中没有的词汇， PAD： 用于PADing文本

    # ------------------------------------- texts -> idxs ----------------------------------------- #
    print('start converting text to indexs...')
    def texts2Indexs(texts):
        total_indexs = []
        for text in texts:
            words = get_paragraph_words(text)
            seq_len = len(words)

            if seq_len < maxSenLen:
                words.extend([vocab_dic.get(PAD)] * (maxSenLen - seq_len))
            elif seq_len > maxSenLen:
                words = words[:maxSenLen]

            # word -> id
            indexs = []
            for word in words:
                indexs.append(vocab_dic.get(word, vocab_dic.get(UNK)))

            total_indexs.append(indexs)
        return total_indexs

    trainTextsIds = texts2Indexs(train_texts)
    testTextsIds = texts2Indexs(test_texts)

    # 转变为numpy arrays
    trainTextsIds = np.array(trainTextsIds)
    train_labels = np.array(train_labels)
    testTextsIds = np.array(testTextsIds)
    test_labels = np.array(test_labels)

    # --------------------------------- 读取词向量 glove.6B.2000d ------------------------------ #
    if embeddingPath == None:
        vectors_dic = {}  # 词向量字典
        with open(r'./pretrained_WordVectors/glove.6B.200d.txt', encoding='utf-8') as file:
            lines = file.readlines()
            print("start constructing the dictionary of Word-Vectors...")
            for line in tqdm(lines):
                line = line.strip().split()
                word = line[0]
                vectors = [float(d) for d in line[1:]]
                vectors_dic[word] = vectors

        # print(len(vectors_dic))
        print("start constructing embedding matrix...")
        # sum = 0
        embedding = np.zeros(shape=(vocabLen + 2, 200), dtype=np.float32)  # 词向量矩阵
        for word, idx in tqdm(vocab_dic.items()):
            # if word not in vectors_dic:
            #     sum = sum + 1
            embedding[idx, :] = vectors_dic.get(word, np.zeros(shape=200, dtype=np.float32))  # 如果存在则取，不存在则用[0,0,...]

        # print(sum)
        np.save('./pretrained_WordVectors/glove.6B.200d.gen.npy', embedding)
    else:
        embedding = np.load(embeddingPath)

    return trainTextsIds, train_labels, testTextsIds, test_labels, embedding


class dataLoader(object):
    '''
    self-defined dataloader
    '''
    def __init__(self, dataset, labels, batchSize, shuffle=True):
        super(dataLoader, self).__init__()

        self.dataset = dataset
        self.labels = labels
        self.batchSize = batchSize
        self.batch_index = 0
        self.shuffle = shuffle

        if self.shuffle:
            order = np.arange(0, self.dataset.shape[0])
            np.random.shuffle(order)
            self.dataset = self.dataset[order]
            self.labels = self.labels[order]

    def next_batch(self):
        start = self.batch_index
        self.batch_index += self.batchSize
        if self.batch_index > self.dataset.shape[0]:
            order = np.arange(0, self.dataset.shape[0])
            np.random.shuffle(order)
            self.dataset = self.dataset[order]
            self.labels = self.labels[order]
            start = 0
            self.batch_index = self.batchSize
        end = self.batch_index
        return self.dataset[start: end], self.labels[start: end]


if __name__=="__main__":

    trainTextsIds, train_labels, testTextsIds, test_labels, embeddings = loadIMDBData(200, 20000)
    print(type(trainTextsIds), trainTextsIds.shape)
    print(trainTextsIds[0:10,:])
    print(type(train_labels), train_labels.shape)
    print(train_labels[13000:13010])
    print(type(testTextsIds), testTextsIds.shape)
    print(testTextsIds[0:10, :])
    print(type(test_labels), test_labels.shape)
    print(test_labels[0:10])
    print(type(embeddings), embeddings.shape)
    print(embeddings[0:10])
    # print(vocab_list[4000][0], vocab_list[4000][1])
    # print(len(vocab_dic))

    trainDataLoader = dataLoader(trainTextsIds, train_labels, 100, shuffle=True)
    testDataLoader = dataLoader(testTextsIds, test_labels, 100, shuffle=False)

    print(trainDataLoader.next_batch())
    print(testDataLoader.next_batch())

    '''
    embedding = {}
    with open(r'./pretrained_WordVectors/glove.6B.200d.txt', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            word = line[0]
            vectors = [float(d) for d in line[1:]]

            print(word)
            print(len(vectors))
            print(vectors)
    '''
