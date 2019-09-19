import os
import torch
import numpy as np
import pandas as pd

try:
    from util import training_loop_text_classification, get_data, emb2size  # noqa
except ImportError:
    from clcd.rnn_train.util import training_loop_text_classification, get_data, emb2size  # noqa


try:
    from RNNConfig import RNNConfig  # noqa
except ImportError:
    from clcd.rnn_train.RNNConfig import RNNConfig  # noqa


try:
    from DataHolder import DataHolder  # noqa
except ImportError:
    from clcd.rnn_train.DataHolder import DataHolder  # noqa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model_on_params(Model,
                          train_data_path,
                          test_data_path,
                          pkl_path,
                          epochs,
                          embedding_dim,
                          layers,
                          rnn_dim,
                          learning_rate,
                          momentum,
                          load_emb=None,
                          bidirectional=False,
                          freeze_emb=False,
                          opt="sgd",
                          drop=0,
                          toy=False):
    """
    Train model on param


    :param model: recurrent model
    :type model: RNN, LSTM, GRU
    :param train_data_path: path to train
    :type train_data_path: str
    :param test_data_path: path to test
    :type test_data_path: str
    :param pkl_path: path to model
    :type pkl_path: str
    :param epochs: number of epochs
    :type epochs: int
    :param embedding_dim: embedding dimention
    :type embedding_dim: int
    :param rnn_dim: rnn hidden size dimention
    :type rnn_dim: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param momentum: momentum param
    :type momentum: float


    :return: accuracy on the valid data
    :rtype: float
    """
    TEXT, LABEL, train, valid, test = get_data(train_data_path,
                                               test_data_path,
                                               toy)

    if load_emb is not None:
        TEXT.vocab.load_vectors(load_emb)
        embedding_dim = emb2size[load_emb]

    current_config = RNNConfig(vocab_size=len(TEXT.vocab),
                               output_dim=len(LABEL.vocab),
                               epochs=epochs,
                               layers=layers,
                               rnn_dim=rnn_dim,
                               embedding_dim=embedding_dim,
                               learning_rate=learning_rate,
                               momentum=momentum,
                               bidirectional=bidirectional,
                               opt=opt,
                               drop=drop)
    model = Model(current_config)

    if load_emb is not None:
        model.embedding.weight.data.copy_(TEXT.vocab.vectors)

    if freeze_emb:
        for param in model.embedding.parameters():
            param.requires_grad = False

    current_data = DataHolder(current_config,
                              train,
                              valid,
                              test)

    training_loop_text_classification(model,
                                      current_config,
                                      current_data,
                                      pkl_path,
                                      verbose=False)

    valid_bach = next(iter(current_data.valid_iter))
    acc, _, _ = model.evaluate_bach(valid_bach, device)

    return acc


def get_prediction_and_text_RNN(Model,
                                train_data_path,
                                test_data_path,
                                out_path_hit,
                                out_path_miss,
                                pkl_path,
                                epochs,
                                embedding_dim,
                                rnn_dim,
                                layers,
                                learning_rate,
                                momentum,
                                bidirectional,
                                toy=False):
    """
    create csv with hit and misses using the model on the test data

    :param model: recurrent model
    :type model: RNN, LSTM, GRU
    :param train_data_path: path to train
    :type train_data_path: str
    :param test_data_path: path to test
    :type test_data_path: str
    :param pkl_path: path to model
    :type pkl_path: str
    :param epochs: number of epochs
    :type epochs: int
    :param embedding_dim: embedding dimention
    :type embedding_dim: int
    :param rnn_dim: rnn hidden size dimention
    :type rnn_dim: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param momentum: momentum param
    :type momentum: float


    :return: accuracy on the valid data
    :rtype: float
    """

    TEXT, LABEL, train, valid, test = get_data(train_data_path,
                                               test_data_path,
                                               toy)
    current_config = RNNConfig(vocab_size=len(TEXT.vocab),
                               output_dim=len(LABEL.vocab),
                               epochs=epochs,
                               rnn_dim=rnn_dim,
                               embedding_dim=embedding_dim,
                               layers=layers,
                               learning_rate=learning_rate,
                               momentum=momentum,
                               bidirectional=bidirectional)

    current_data = DataHolder(current_config,
                              train,
                              valid,
                              test)

    model = Model(current_config)
    model.load_state_dict(torch.load(pkl_path))
    model.to(device)
    test_batch = next(iter(current_data.test_iter))
    all_text = [" ".join([TEXT.vocab.itos[k] for k in i])
                for i in test_batch.text.t().numpy()]
    batch_text = test_batch.text.to(device)
    pred = model.predict(batch_text)
    label = test_batch.label
    pred = pred.cpu().numpy()
    label = label.cpu().numpy().astype(int)
    df = pd.DataFrame({"text": all_text,
                       "pred": pred,
                       "label": label})
    df_hit = df.loc[pred == label]
    df_miss = df.loc[pred != label]
    df_hit.to_csv(out_path_hit, index=False)
    df_miss.to_csv(out_path_miss, index=False)


def eval_model_on_test(Model,
                       train_data_path,
                       test_data_path,
                       pkl_path,
                       epochs,
                       embedding_dim,
                       rnn_dim,
                       layers,
                       learning_rate,
                       momentum,
                       bidirectional,
                       toy=False):
    """
    Eval model on param

    :param model: recurrent model
    :type model: RNN, LSTM, GRU
    :param train_data_path: path to train
    :type train_data_path: str
    :param test_data_path: path to test
    :type test_data_path: str
    :param pkl_path: path to model
    :type pkl_path: str
    :param epochs: number of epochs
    :type epochs: int
    :param embedding_dim: embedding dimention
    :type embedding_dim: int
    :param rnn_dim: rnn hidden size dimention
    :type rnn_dim: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param momentum: momentum param
    :type momentum: float


    :return: accuracy on the valid data
    :rtype: float
    """

    TEXT, LABEL, train, valid, test = get_data(train_data_path,
                                               test_data_path,
                                               toy)
    current_config = RNNConfig(vocab_size=len(TEXT.vocab),
                               output_dim=len(LABEL.vocab),
                               epochs=epochs,
                               rnn_dim=rnn_dim,
                               embedding_dim=embedding_dim,
                               layers=layers,
                               learning_rate=learning_rate,
                               momentum=momentum,
                               bidirectional=bidirectional)

    current_data = DataHolder(current_config,
                              train,
                              valid,
                              test)

    model = Model(current_config)
    model.load_state_dict(torch.load(pkl_path))
    model.to(device)
    test_bach = next(iter(current_data.test_iter))
    acc, _, _ = model.evaluate_bach(test_bach, device)

    return acc


def get_random_discrete_param(lower_bound, upper_bound):

    return np.random.randint(lower_bound, upper_bound)


def get_random_cont_param(lower_bound=0, upper_bound=1):

    return np.random.uniform(lower_bound, upper_bound)


def get_random_from_list(list_):

    return np.random.choice(list_)


def random_search(Model,
                  trials,
                  train_data_path,
                  test_data_path,
                  epoch_bounds=[1, 10],
                  embedding_dim_bounds=[10, 500],
                  rnn_dim_bounds=[10, 500],
                  layers_bounds=[1, 6],
                  learning_rate_bounds=[0, 1],
                  momentum_bounds=[0, 1],
                  verbose=True,
                  prefix="",
                  acc_bound=1.0,
                  load_emb=None,
                  bidirectional=False,
                  freeze_emb=False,
                  opt="sgd",
                  drop_bounds=[0, 1],
                  toy=False,
                  base_path="tmp_pkl"):
    """
    Train model in n trails on random params


    :param model: recurrent model
    :type model: RNN, LSTM, GRU
    :param train_data_path: path to train
    :type train_data_path: str
    :param test_data_path: path to test
    :type test_data_path: str
    :param pkl_path: path to model
    :type pkl_path: str
    :param epochs: number of epochs
    :type epochs: int
    :param embedding_dim: embedding dimention
    :type embedding_dim: int
    :param rnn_dim: rnn hidden size dimention
    :type rnn_dim: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param momentum: momentum param
    :type momentum: float
    :param acc_bound: upper bound to control how many trails will be performed
    :type acc_bound: float


    :return: list  of accuracy, list of hyperperams, list of pkl paths
    :rtype: [float], [dict], [str]
    """

    all_acc = []
    all_hyper_params = []
    all_names = []
    best_acc = 0
    for i in range(trials):
        if best_acc < acc_bound:
            if verbose:
                print("=== random search: ({}/{})\n".format(i + 1, trials))
            epochs = get_random_discrete_param(epoch_bounds[0], epoch_bounds[1])  # noqa
            embedding_dim = get_random_discrete_param(embedding_dim_bounds[0],
                                                      embedding_dim_bounds[1])
            rnn_dim = get_random_discrete_param(rnn_dim_bounds[0],
                                                rnn_dim_bounds[1])

            layers = get_random_discrete_param(layers_bounds[0],
                                               layers_bounds[1])

            learning_rate = get_random_cont_param(learning_rate_bounds[0],
                                                  learning_rate_bounds[1])

            momentum = get_random_cont_param(momentum_bounds[0],
                                             momentum_bounds[1])

            drop = get_random_cont_param(drop_bounds[0],
                                         drop_bounds[1])

            if load_emb is not None:
                embedding_dim = emb2size[load_emb]

            hyper_dict = {"epochs": epochs,
                          "embedding_dim": embedding_dim,
                          "rnn_dim": rnn_dim,
                          "layers": layers,
                          "learning_rate": learning_rate,
                          "momentum": momentum,
                          "load_emb": load_emb,
                          "bidirectional": bidirectional,
                          "freeze_emb": freeze_emb,
                          "opt": opt,
                          "drop": drop,
                          "model": Model.name}

            if not os.path.exists(base_path):
                os.makedirs(base_path)

            name = "{}_embedding_{}_epochs_{}_layers_{}_embedding_dim_{}_rnn_dim_{}_learning_rate_{:.3f}_momentum_{:.3f}_bi_{}_freeze_emb_{}_opt_{}_drop_{:.3f}".format(hyper_dict["model"],  # noqa
                                                                                                                                                                        hyper_dict["load_emb"],  # noqa
                                                                                                                                                                        hyper_dict["epochs"],  # noqa
                                                                                                                                                                        hyper_dict["layers"],  # noqa
                                                                                                                                                                        hyper_dict["embedding_dim"],  # noqa
                                                                                                                                                                        hyper_dict["rnn_dim"],  # noqa
                                                                                                                                                                        hyper_dict["learning_rate"],  # noqa
                                                                                                                                                                        hyper_dict["momentum"],  # noqa
                                                                                                                                                                        hyper_dict["bidirectional"],  # noqa
                                                                                                                                                                        hyper_dict["freeze_emb"],  # noqa
                                                                                                                                                                        hyper_dict["opt"],  # noqa
                                                                                                                                                                        hyper_dict["drop"])  # noqa
            name = name.replace(".", "p") + ".pkl"
            name = os.path.join(base_path, prefix + name)

            acc = train_model_on_params(Model=Model,
                                        train_data_path=train_data_path,
                                        test_data_path=test_data_path,
                                        pkl_path=name,
                                        epochs=epochs,
                                        embedding_dim=embedding_dim,
                                        layers=layers,
                                        rnn_dim=rnn_dim,
                                        learning_rate=learning_rate,
                                        momentum=momentum,
                                        load_emb=load_emb,
                                        bidirectional=bidirectional,
                                        freeze_emb=freeze_emb,
                                        opt=opt,
                                        drop=drop,
                                        toy=toy)
            if verbose:
                print("====== dict", hyper_dict)
                print("====== acc", acc)
            all_acc.append(acc)
            all_hyper_params.append(hyper_dict)
            all_names.append(name)
            best_acc = acc

    return all_acc, all_hyper_params, all_names


def naive_grid_search(Model,
                      search_trials,
                      random_trials,
                      train_data_path,
                      test_data_path,
                      epoch_bounds=[1, 10],
                      embedding_dim_bounds=[10, 500],
                      rnn_dim_bounds=[10, 500],
                      layers_bounds=[1, 6],
                      learning_rate_bounds=[0, 1],
                      momentum_bounds=[0, 1],
                      verbose=True,
                      prefix="",
                      acc_bound=1.0,
                      load_emb=None,
                      bidirectional=False,
                      freeze_emb=False,
                      opt="sgd",
                      drop_bounds=[0, 1],
                      toy=False,
                      base_path="tmp_pkl"):
    """
    Train model using random params, at each time in search_trials
    the hyper param search is reduce. At the end, the best model
    (with the best accuracy on the valid dataset is select) is returned


    :param model: recurrent model
    :type model: RNN, LSTM, GRU
    :param train_data_path: path to train
    :type train_data_path: str
    :param test_data_path: path to test
    :type test_data_path: str
    :param pkl_path: path to model
    :type pkl_path: str
    :param epochs: number of epochs
    :type epochs: int
    :param embedding_dim: embedding dimention
    :type embedding_dim: int
    :param rnn_dim: rnn hidden size dimention
    :type rnn_dim: int
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param momentum: momentum param
    :type momentum: float
    :param acc_bound: upper bound to control how many trails will be performed
    :type acc_bound: float


    :return: test accuracy, hyperperams, pkl path
    :rtype: float, dict, str
    """
    epoch_bounds = epoch_bounds
    embedding_dim_bounds = embedding_dim_bounds
    learning_rate_bounds = learning_rate_bounds
    momentum_bounds = momentum_bounds
    rnn_dim_bounds = rnn_dim_bounds
    layers_bounds = layers_bounds
    drop_bounds = drop_bounds
    best_acc = 0
    best_params = None
    model_path = None

    for i in range(search_trials):
        if best_acc < acc_bound:
            if verbose:
                print("grid_search ({}/{})\n".format(i + 1, search_trials))

            all_acc, all_hyper_params, all_names = random_search(Model=Model,  # noqa
                                                                 trials=random_trials,  # noqa
                                                                 train_data_path=train_data_path,  # noqa
                                                                 test_data_path=test_data_path,  # noqa
                                                                 epoch_bounds=epoch_bounds,  # noqa
                                                                 embedding_dim_bounds=embedding_dim_bounds,  # noqa
                                                                 rnn_dim_bounds=rnn_dim_bounds,  # noqa
                                                                 layers_bounds=layers_bounds,  # noqa
                                                                 learning_rate_bounds=learning_rate_bounds,  # noqa
                                                                 momentum_bounds=momentum_bounds,  # noqa
                                                                 verbose=verbose,  # noqa
                                                                 prefix=prefix,  # noqa
                                                                 acc_bound=acc_bound,  # noqa
                                                                 load_emb=load_emb,  # noqa
                                                                 bidirectional=bidirectional,  # noqa
                                                                 freeze_emb=freeze_emb,  # noqa
                                                                 opt=opt,
                                                                 drop_bounds=drop_bounds,  # noqa
                                                                 toy=toy,
                                                                 base_path=base_path)  # noqa

            best_i = np.argmax(all_acc)
            current_acc = all_acc[best_i]  # noqa
            current_dict = all_hyper_params[best_i]  # noqa
            name = all_names[best_i]
            if best_acc < current_acc:
                epoch_bounds = [epoch_bounds[0],
                                current_dict["epochs"] + 1]

                embedding_dim_bounds = [embedding_dim_bounds[0],
                                        current_dict["embedding_dim"] + 1]

                rnn_dim_bounds = [rnn_dim_bounds[0],
                                  current_dict["rnn_dim"] + 1]

                layers_bounds = [layers_bounds[0],
                                 current_dict["layers"] + 1]

                learning_rate_bounds = [learning_rate_bounds[0],
                                        current_dict["learning_rate"]]

                momentum_bounds = [momentum_bounds[0],
                                   current_dict["momentum"]]

                drop_bounds = [drop_bounds[0],
                               current_dict["momentum"]]

                best_acc = current_acc
                best_params = current_dict
                model_path = name

    test_acc = eval_model_on_test(Model=Model,
                                  train_data_path=train_data_path,
                                  test_data_path=test_data_path,
                                  pkl_path=model_path,
                                  epochs=best_params["epochs"],
                                  embedding_dim=best_params["embedding_dim"],  # noqa
                                  layers=best_params["layers"],
                                  rnn_dim=best_params["rnn_dim"],  # noqa
                                  learning_rate=best_params["learning_rate"],  # noqa
                                  momentum=best_params["momentum"],
                                  bidirectional=best_params["bidirectional"],
                                  toy=toy)

    return test_acc, best_params, model_path
