# TextCNN & TextRNN

This is an implementation of  TextCNN and TextRNN of PyTorch version.   

some codes are referred to https://github.com/shark803/Text-Classification-Pytorch

## Dataset & pretrained word-vectors

IMDB dataset are chose to do the text classification task.  You can download it on http://ai.stanford.edu/~amaas/data/sentiment/   

Glove.2B.200d are used as the pretrained word-vectors. You can download it on https://nlp.stanford.edu/projects/glove/

## Usage

- Run CNN Model

```
python train_test.py --model CNN
```

- Run RNN Model

```
python train_test.py --model LSTM
```

