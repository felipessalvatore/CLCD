import matplotlib
matplotlib.use('Agg')
import torch  # noqa
import torch.nn as nn  # noqa
import torch.optim as optim  # noqa
import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
import pandas as pd  # noqa
from torchtext import data  # noqa
import os  # noqa
import inspect  # noqa
import sys  # noqa
import time  # noqa
import math  # noqa


almost_current = os.path.abspath(inspect.getfile(inspect.currentframe()))
currentdir = os.path.dirname(almost_current)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from text_processing.functions import simple_pre_process_text_df  # noqa
except ImportError:
    from clcd.text_processing.functions import simple_pre_process_text_df  # noqa


def pre_process_df(train_data, test_data):
    """
    Transform the dataframes train_data and
    test_data using the simple_pre_process_text_df
    fuction


    :param train_data: train dataset
    :type train_data: pd.DataFrame
    :param test_data: test dataset
    :type test_data: pd.DataFrame
    :return: transformed train dataset, transformed test dataset
    :rtype: pd.DataFrame, pd.DataFrame
    """
    train_data["text"] = train_data["sentence1"] + ", " + train_data["sentence2"]  # noqa
    test_data["text"] = test_data["sentence1"] + ", " + test_data["sentence2"]
    train_data.drop(["sentence1", "sentence2"], axis=1, inplace=True)
    test_data.drop(["sentence1", "sentence2"], axis=1, inplace=True)
    train_data = train_data[["text", "label"]]
    test_data = test_data[["text", "label"]]
    simple_pre_process_text_df(train_data)
    simple_pre_process_text_df(test_data)
    return train_data, test_data


def pre_process_df_and(train_data, test_data):
    """
    Transform the dataframes train_data and
    test_data using the simple_pre_process_text_df
    fuction


    :param train_data: train dataset
    :type train_data: pd.DataFrame
    :param test_data: test dataset
    :type test_data: pd.DataFrame
    :return: transformed train dataset, transformed test dataset
    :rtype: pd.DataFrame, pd.DataFrame
    """
    train_data["text"] = train_data["sentence1"] + ", " + train_data["sentence2"]  # noqa
    test_data["text"] = test_data["sentence1"] + ", " + test_data["sentence2"]
    train_data.drop(["sentence1", "sentence2"], axis=1, inplace=True)
    test_data.drop(["sentence1", "sentence2"], axis=1, inplace=True)
    train_data = train_data[["text", "label"]]
    test_data = test_data[["text", "label"]]
    simple_pre_process_text_df(train_data)
    simple_pre_process_text_df(test_data)
    return train_data, test_data


def train_in_epoch(model, iterator, optimizer, criterion, negative=False):
    """
    Train the model using all the data from the iterator

    Negative is a flag to train the model to maximize the
    loss function

    :param model: RNN classification model
    :type model: RNN
    :param iterator: data iterator
    :type iterator: data.BucketIterator
    :param optimizer: torch optimizer
    :type optimizer: optim.SGD, etc.
    :param criterion: loss criterion
    :type criterion: nn.CrossEntropyLoss
    :param negative: control param
    :type negative: boolean
    :return: mean loss, mean accuracy
    :rtype: float, float
    """

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        label = batch.label.type("torch.LongTensor")
        label = label.to(device)
        batch_text = batch.text.to(device)
        logits = model(batch_text)
        loss = criterion(logits, label)
        if negative:
            loss = - loss
        acc, _, _ = model.evaluate_bach(batch, device)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    mean_loss = epoch_loss / len(iterator)
    mean_acc = epoch_acc / len(iterator)

    return mean_loss, mean_acc


def get_valid_loss(model, valid_iter, criterion):
    """
    Get the valid loss

    :param model: RNN classification model
    :type model:
    :param valid_iter: valid iterator
    :type valid_iter: data.BucketIterator
    :param criterion: loss criterion
    :type criterion: nn.CrossEntropyLoss
    :return: valid loss
    :rtype: Tensor(shape=[])
    """
    batch = next(iter(valid_iter))
    model.eval()
    label = batch.label.type("torch.LongTensor")
    label = label.to(device)
    batch_text = batch.text.to(device)
    logits = model(batch_text)
    loss = criterion(logits, label)
    return loss


def training_loop_text_classification(model,
                                      config,
                                      dataholder,
                                      model_path,
                                      verbose=True,
                                      negative=False,
                                      plot_path="training plot.png"):
    """
    Train a model for text classification

    :param model: RNN classification model
    :type model: RNN
    :param config: classification model config
    :type config: RNNConfig
    :param dataholder: data
    :type dataholder: DataHolder or DataHolderGentle
    :param model_path: path to save model params
    :type model_path: str
    :param verbose: param to control print
    :type verbose: bool
    """
    if config.opt == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=config.learning_rate)

    elif config.opt == "adagrad":
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=config.learning_rate)

    elif config.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=config.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=config.learning_rate,
                              momentum=config.momentum)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    train_iter = dataholder.train_iter
    valid_iter = dataholder.valid_iter

    best_valid_loss = float("inf")

    all_train_loss = []
    all_valid_loss = []

    for epoch in range(config.epochs):

        if verbose:
            print("epoch = ({}/{})".format(epoch + 1, config.epochs))
        train_loss, train_acc = train_in_epoch(model,
                                               train_iter,
                                               optimizer,
                                               criterion,
                                               negative)
        all_train_loss.append(train_loss)
        valid_loss = get_valid_loss(model, valid_iter, criterion)
        all_valid_loss.append(valid_loss)

        msg = "\ntrain_loss = {:.3f} | valid_loss = {:.3f}".format(float(train_loss),  # noqa
                                                                   float(valid_loss))  # noqa

        if float(valid_loss) < best_valid_loss:
            torch.save(model.state_dict(), model_path)
            best_valid_loss = float(valid_loss)

        if verbose:
            print(msg)
            print("train_acc = {}\n".format(train_acc))

    if verbose:
        x = np.arange(1, len(all_train_loss) + 1, 1)
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(x, all_train_loss, label='mean train loss')
        ax.plot(x, all_valid_loss, label='mean valid loss')
        ax.legend()
        plt.xlabel('epoch')
        plt.ylabel('mean loss')
        plt.title('Train and valid mean loss')
        plt.grid(True)
        plt.savefig(plot_path)


def get_data(train_path,
             test_path,
             tokenize='spacy',
             max_vocab_size=25000,
             train_valid_split=0.8,
             toy=False):
    """
    Transform a csv into data.TabularDataset

    :param train_path: path to train dataset
    :type train_path: str
    :param test_path: path to test dataset
    :type test_path: str
    :param tokenize: tokenize param
    :type tokenize: str
    :param max_vocab_size: vocab size
    :type max_vocab_size: int
    :param train_valid_split: param to control the split between train
                              and valid dataset
    :type train_valid_split: float
    :return: text field, label field, train data, valid data, test data
    :type verbose: data.Field, data.LabelField, data.TabularDataset,
                   data.TabularDataset, data.TabularDataset
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if toy:
        train_data = train_data.head(100)
        test_data = test_data.head(100)

    train_data, test_data = pre_process_df(train_data, test_data)

    train_data_path = "train_processed.csv"
    test_data_path = "test_processed.csv"

    train_data.to_csv(train_data_path, header=False, index=False)
    test_data.to_csv(test_data_path, header=False, index=False)

    if tokenize == 'spacy':
        TEXT = data.Field(tokenize=tokenize)
    else:
        TEXT = data.Field()

    LABEL = data.LabelField(dtype=torch.float)
    train = data.TabularDataset(path=train_data_path,
                                format="csv",
                                fields=[('text', TEXT),
                                        ('label', LABEL)])
    test = data.TabularDataset(path=test_data_path,
                               format="csv",
                               fields=[('text', TEXT),
                                       ('label', LABEL)])

    os.remove(train_data_path)
    os.remove(test_data_path)

    train, valid = train.split(train_valid_split)

    TEXT.build_vocab(train, max_size=max_vocab_size)
    LABEL.build_vocab(train)

    return TEXT, LABEL, train, valid, test


emb_list = ["charngram.100d", "fasttext.en.300d",
            "fasttext.simple.300d",
            "glove.42B.300d",
            "glove.840B.300d",
            "glove.twitter.27B.25d",
            "glove.twitter.27B.50d",
            "glove.twitter.27B.100d",
            "glove.twitter.27B.200d",
            "glove.6B.50d",
            "glove.6B.100d",
            "glove.6B.200d",
            "glove.6B.300d"]

emb2size = {"charngram.100d": 100,
            "fasttext.en.300d": 300,
            "fasttext.simple.300d": 300,
            "glove.42B.300d": 300,
            "glove.840B.300d": 300,
            "glove.twitter.27B.25d": 25,
            "glove.twitter.27B.50d": 50,
            "glove.twitter.27B.100d": 100,
            "glove.twitter.27B.200d": 200,
            "glove.6B.50d": 50,
            "glove.6B.100d": 100,
            "glove.6B.200d": 200,
            "glove.6B.300d": 300}


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % asMinutes(s)
