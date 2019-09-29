import numpy as np
import torch
import time
import unittest
import os
import shutil
from clcd.rnn_train.util import get_data, training_loop_text_classification  # noqa
from clcd.rnn_train.RNNConfig import RNNConfig
from clcd.rnn_train.RNN import RNN
from clcd.rnn_train.LSTM import LSTM
from clcd.rnn_train.GRU import GRU
from clcd.rnn_train.DataHolder import DataHolder
from clcd.plots.functions import plot_confusion_matrix
from clcd.rnn_train.random_search import train_model_on_params
from clcd.rnn_train.random_search import random_search
from clcd.rnn_train.random_search import naive_grid_search

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainFunctionsTest(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        if os.path.exists("tmp_pkl"):
            shutil.rmtree("tmp_pkl")
        if os.path.exists("testRNN.pkl"):
            os.remove("testRNN.pkl")
        if os.path.exists("testRNN.png"):
            os.remove("testRNN.png")
        if os.path.exists("testLSTM.pkl"):
            os.remove("testLSTM.pkl")
        if os.path.exists("testLSTM.png"):
            os.remove("testLSTM.png")
        if os.path.exists("testGRU.pkl"):
            os.remove("testGRU.pkl")
        if os.path.exists("testGRU.png"):
            os.remove("testGRU.png")

    @classmethod
    def setUp(cls):
        cls.path_train = os.path.join("text_gen_output",
                                      "boolean1_train.csv")
        cls.path_test = os.path.join("text_gen_output",
                                     "boolean1_test.csv")
        TEXT, LABEL, train, valid, test = get_data(cls.path_train,
                                                   cls.path_test)

        cls.current_config = RNNConfig(vocab_size=len(TEXT.vocab),
                                       output_dim=len(LABEL.vocab),
                                       epochs=5,
                                       embedding_dim=100,
                                       learning_rate=0.05,
                                       momentum=0.1)

        cls.current_data = DataHolder(cls.current_config,
                                      train,
                                      valid,
                                      test)

    def test_basic_training_RNN(self):
        model = RNN(self.current_config)

        training_loop_text_classification(model,
                                          self.current_config,
                                          self.current_data,
                                          "testRNN.pkl",
                                          verbose=False)
        model = RNN(self.current_config)
        valid_batch = next(iter(self.current_data.valid_iter))
        model.load_state_dict(torch.load("testRNN.pkl"))
        model.to(device)
        acc, pred, labels = model.evaluate_bach(valid_batch, device)
        self.assertTrue(acc > 0.6,
                        "after training, valid_acc = {:.3f} < 0.6".format(acc))
        msg = "problems with the confusion matrix plot"
        labels_legend = ['no', 'yes']
        plot_confusion_matrix(truth=labels.cpu().numpy(),
                              predictions=pred.cpu().numpy(),
                              save=True,
                              path="testRNN.png",
                              classes=labels_legend)
        self.assertTrue(os.path.exists("testRNN.png"), msg=msg)

    def test_basic_training_LSTM(self):
        model = LSTM(self.current_config)
        training_loop_text_classification(model,
                                          self.current_config,
                                          self.current_data,
                                          "testLSTM.pkl",
                                          verbose=False)
        model = LSTM(self.current_config)
        valid_batch = next(iter(self.current_data.valid_iter))
        model.load_state_dict(torch.load("testLSTM.pkl"))
        valid_batch = next(iter(self.current_data.valid_iter))
        model.load_state_dict(torch.load("testLSTM.pkl"))
        model.to(device)
        acc, pred, labels = model.evaluate_bach(valid_batch, device)
        self.assertTrue(acc > 0.7,
                        "after training, valid_acc = {:.3f}".format(acc))
        msg = "problems with the confusion matrix plot"
        labels_legend = ['no', 'yes']
        plot_confusion_matrix(truth=labels.cpu().numpy(),
                              predictions=pred.cpu().numpy(),
                              save=True,
                              path="testLSTM.png",
                              classes=labels_legend)
        self.assertTrue(os.path.exists("testLSTM.png"), msg=msg)

    def test_basic_training_GRU(self):
        model = GRU(self.current_config)
        training_loop_text_classification(model,
                                          self.current_config,
                                          self.current_data,
                                          "testGRU.pkl",
                                          verbose=False)
        model = GRU(self.current_config)
        valid_batch = next(iter(self.current_data.valid_iter))
        model.load_state_dict(torch.load("testGRU.pkl"))
        valid_batch = next(iter(self.current_data.valid_iter))
        model.load_state_dict(torch.load("testGRU.pkl"))
        model.to(device)
        acc, pred, labels = model.evaluate_bach(valid_batch, device)
        self.assertTrue(acc > 0.7,
                        "after training, valid_acc = {:.3f}".format(acc))
        msg = "problems with the confusion matrix plot"
        labels_legend = ['no', 'yes']
        plot_confusion_matrix(truth=labels.cpu().numpy(),
                              predictions=pred.cpu().numpy(),
                              save=True,
                              path="testGRU.png",
                              classes=labels_legend)
        self.assertTrue(os.path.exists("testGRU.png"), msg=msg)

    def test_random_param_train(self):

        acc1 = train_model_on_params(Model=RNN,
                                     train_data_path=self.path_train,
                                     test_data_path=self.path_test,
                                     pkl_path="testRNN.png",
                                     epochs=5,
                                     embedding_dim=100,
                                     layers=1,
                                     rnn_dim=100,
                                     learning_rate=0.05,
                                     momentum=0.1)

        acc2 = train_model_on_params(Model=GRU,
                                     train_data_path=self.path_train,
                                     test_data_path=self.path_test,
                                     pkl_path="testRNN.png",
                                     epochs=5,
                                     embedding_dim=50,
                                     layers=1,
                                     rnn_dim=50,
                                     learning_rate=0.05,
                                     momentum=0.1)

        acc3 = train_model_on_params(Model=LSTM,
                                     train_data_path=self.path_train,
                                     test_data_path=self.path_test,
                                     pkl_path="testRNN.png",
                                     epochs=5,
                                     embedding_dim=23,
                                     rnn_dim=30,
                                     layers=1,
                                     learning_rate=0.05,
                                     momentum=0.1)
        acc = acc1 + acc2 + acc3
        msg = "after training, valid_acc = {:.3f} < {:.3f}".format(acc, 0.6 * 3)  # noqa
        self.assertTrue(acc >= 0.57 * 3, msg=msg)

    def test_random_param_train_bound(self):

        all_acc, all_hyper_params, all_names = random_search(Model=RNN,
                                                             trials=10,
                                                             train_data_path=self.path_train,  # noqa
                                                             test_data_path=self.path_test,  # noqa
                                                             epoch_bounds=[1, 2],  # noqa
                                                             embedding_dim_bounds=[10, 500],  # noqa
                                                             rnn_dim_bounds=[10, 500],  # noqa
                                                             layers_bounds=[1, 2],  # noqa
                                                             learning_rate_bounds=[0, 1],  # noqa
                                                             momentum_bounds=[0, 1],  # noqa
                                                             verbose=False,  # noqa
                                                             prefix="RNN_boolean1_",  # noqa
                                                             acc_bound=0.6)  # noqa
        cond = len(all_acc) == len(all_hyper_params) == len(all_names)
        cond_bound = len(all_acc) < 10
        self.assertTrue(cond, msg="different output sizes")
        self.assertTrue(cond_bound,
                        msg="not stoping, len(all_acc) = {}".format(len(all_acc)))  # noqa

    def test_grid_search_bound(self):

        init = time.time()
        _, _, _ = naive_grid_search(Model=RNN,
                                    search_trials=1,
                                    random_trials=1,
                                    train_data_path=self.path_train,
                                    test_data_path=self.path_test,
                                    epoch_bounds=[1, 2],
                                    layers_bounds=[1, 2],
                                    verbose=False,
                                    prefix="RNN_boolean1_")
        reference = time.time() - init

        init = time.time()

        test_accRNN, _, _ = naive_grid_search(Model=RNN,
                                              search_trials=10,
                                              random_trials=10,
                                              train_data_path=self.path_train,
                                              test_data_path=self.path_test,
                                              epoch_bounds=[1, 2],
                                              layers_bounds=[1, 2],
                                              verbose=False,
                                              prefix="RNN_boolean1_",
                                              acc_bound=0.5)
        experiment = time.time() - init
        msg = "taking too much time, ref ={:.3f}, exp ={:.3f}".format(reference, experiment)  # noqa
        cond = experiment <= 3 * reference
        self.assertTrue(cond, msg=msg)

    def test_random_search(self):
        all_acc, all_hyper_params, all_names = random_search(Model=RNN,
                                                             trials=2,
                                                             train_data_path=self.path_train,  # noqa
                                                             test_data_path=self.path_test,  # noqa
                                                             epoch_bounds=[1, 2],  # noqa
                                                             embedding_dim_bounds=[10, 500],  # noqa
                                                             rnn_dim_bounds=[10, 500],  # noqa
                                                             learning_rate_bounds=[0, 1],  # noqa
                                                             layers_bounds=[1, 2],  # noqa
                                                             momentum_bounds=[0, 1],  # noqa
                                                             verbose=False,  # noqa
                                                             prefix="RNN_boolean1_")  # noqa
        cond = len(all_acc) == len(all_hyper_params) == len(all_names)
        self.assertTrue(cond, msg="different output sizes")
        self.assertTrue(np.max(all_acc) > 0.56,
                        msg="acc list = {}".format(all_acc))

    def test_naive_grid_search(self):
        test_accRNN, _, _ = naive_grid_search(Model=RNN,
                                              search_trials=2,
                                              random_trials=2,
                                              train_data_path=self.path_train,
                                              test_data_path=self.path_test,
                                              epoch_bounds=[1, 2],
                                              layers_bounds=[1, 2],
                                              verbose=False,
                                              prefix="RNN_boolean1_")
        test_accGRU, _, _ = naive_grid_search(Model=GRU,
                                              search_trials=2,
                                              random_trials=2,
                                              train_data_path=self.path_train,
                                              test_data_path=self.path_test,
                                              epoch_bounds=[1, 2],
                                              layers_bounds=[1, 2],
                                              verbose=False,
                                              prefix="GRU_boolean1_")

        test_accLSTM, _, _ = naive_grid_search(Model=LSTM,
                                               search_trials=2,
                                               random_trials=2,
                                               train_data_path=self.path_train,
                                               test_data_path=self.path_test,
                                               epoch_bounds=[1, 2],
                                               layers_bounds=[1, 2],
                                               verbose=False,
                                               prefix="LSTM_boolean1_")
        acc = test_accRNN + test_accGRU + test_accLSTM
        msg = "after training, valid_acc = {:.3f}, {:.3f}, {:.3f}".format(test_accRNN,  # noqa
                                                                          test_accGRU,  # noqa
                                                                          test_accLSTM)  # noqa
        self.assertTrue(acc >= 0.6 * 3, msg=msg)
