import pandas as pd
import os
import argparse
from clcd.rnn_train.RNN import RNN
from clcd.rnn_train.LSTM import LSTM
from clcd.rnn_train.GRU import GRU
from clcd.rnn_train.random_search import naive_grid_search
from clcd.rnn_train.random_search import get_prediction_and_text_RNN
from clcd.bert_train.classification import ContraProcessor, eval_model
from clcd.bert_train.classification import ExplicitProcessor
from clcd.text_processing.functions import simple_pre_process_text_df
from clcd.text_processing.functions import train_bow_xgb_classifier
from clcd.text_processing.functions import describe_data
from clcd.bert_train.classification import get_training_obj, train_model


def train_RNN(train_path_,
              test_path_,
              base_path,
              prefix,
              search_trials,
              random_trials,
              load_emb,
              bidirectional,
              freeze_emb,
              opt,
              Model,
              toy):
    base_path_models = os.path.join(base_path, "rnn_models")
    base_path_outputs = os.path.join(base_path, "rnn_outputs")
    best_acc, best_params, name = naive_grid_search(Model,
                                                    search_trials=search_trials,  # noqa
                                                    random_trials=random_trials,  # noqa
                                                    train_data_path=train_path_,  # noqa
                                                    test_data_path=test_path_,
                                                    epoch_bounds=[1, 10],
                                                    embedding_dim_bounds=[
                                                        10, 500],
                                                    rnn_dim_bounds=[10, 500],
                                                    layers_bounds=[1, 6],
                                                    learning_rate_bounds=[
                                                        0, 1],
                                                    momentum_bounds=[0, 1],
                                                    verbose=True,
                                                    prefix=prefix,
                                                    acc_bound=1.0,
                                                    load_emb=load_emb,
                                                    bidirectional=bidirectional,  # noqa
                                                    freeze_emb=freeze_emb,
                                                    opt=opt,
                                                    drop_bounds=[0, 1],
                                                    toy=toy,
                                                    base_path=base_path_models)  # noqa

    if not os.path.exists(base_path_outputs):
        os.makedirs(base_path_outputs)

    name_ = name.split("/")[-1]
    name_ = name_.split(".")[0]

    write_path = os.path.join(base_path_outputs, name_ + "_results.txt")  # noqa
    with open(write_path, "w") as file:
        file.write("acc =  {:.3f}\n".format(best_acc))
        file.write("best_params =  {}\n".format(best_params))
        file.write("model path =  {}\n".format(name))

    get_prediction_and_text_RNN(Model,
                                train_data_path=train_path_,
                                test_data_path=test_path_,
                                out_path_hit=os.path.join(base_path_outputs, "hit.txt"),  # noqa
                                out_path_miss=os.path.join(base_path_outputs, "miss.txt"),  # noqa
                                pkl_path=name,
                                epochs=best_params['epochs'],
                                embedding_dim=best_params['embedding_dim'],
                                rnn_dim=best_params['rnn_dim'],
                                layers=best_params['layers'],
                                learning_rate=best_params['learning_rate'],
                                momentum=best_params['momentum'],
                                bidirectional=best_params['bidirectional'],
                                toy=toy)


def train_BERT(train_path_,
               test_path_,
               base_path_,
               prefix,
               args_,
               processors_={"contraboolean": ContraProcessor,
                            "explicit": ExplicitProcessor},
               num_labels_task_={"contraboolean": 2,
                                 "explicit": 6}):

    if not os.path.exists(base_path_):
        os.makedirs(base_path_)

    base_path_models = os.path.join(base_path_, "BERT_models")
    base_path_outputs = os.path.join(base_path_, "BERT_outputs")

    df_train = pd.read_csv(train_path_)
    df_dev = pd.read_csv(test_path_)
    simple_pre_process_text_df(df_dev, key="sentence1")
    simple_pre_process_text_df(df_dev, key="sentence2")
    simple_pre_process_text_df(df_train, key="sentence1")
    simple_pre_process_text_df(df_train, key="sentence2")
    if args_["toy"]:

        df_train = df_train.head(100)
        df_dev = df_dev.head(100)

    obj_dict = get_training_obj(args_=args_,
                                df_train_=df_train,
                                df_dev_=df_dev,
                                processors_=processors_,
                                num_labels_task_=num_labels_task_)

    args = obj_dict["args"]
    model = obj_dict["model"]
    train_examples = obj_dict["train_examples"]
    eval_examples = obj_dict["eval_examples"]
    optimizer = obj_dict["optimizer"]
    tokenizer = obj_dict["tokenizer"]
    logger = obj_dict["logger"]
    num_train_steps = obj_dict["num_train_steps"]
    t_total = obj_dict["t_total"]
    device = obj_dict["device"]
    n_gpu = obj_dict["n_gpu"]
    label_list = obj_dict["label_list"]
    num_labels = obj_dict["num_labels"]

    path, loss, global_step = train_model(model_=model,
                                          model_name_=prefix,
                                          train_examples_=train_examples,
                                          label_list_=label_list,
                                          args_=args,
                                          tokenizer_=tokenizer,
                                          logger_=logger,
                                          num_train_steps_=num_train_steps,
                                          optimizer_=optimizer,
                                          n_gpu_=n_gpu,
                                          t_total_=t_total,
                                          device_=device,
                                          base_path_model=base_path_models)

    _ = eval_model(output_model_file_=path,  # noqa
                   model_name_=prefix,
                   eval_examples_=eval_examples,
                   args_=args,
                   logger_=logger,
                   tokenizer_=tokenizer,
                   device_=device,
                   num_labels_=num_labels,
                   label_list_=label_list,
                   loss_=loss,
                   global_step_=global_step,
                   base_path_output=base_path_outputs)


def train(train_path_,
          test_path_,
          name_,
          args_,
          search_trials,
          random_trials,
          load_emb,
          bidirectional,
          freeze_emb,
          opt,
          Model,
          toy,
          no_rnn_,
          no_bert_,
          no_baseline_,
          num_words_,
          reduce_str_,
          no_describe):

    if not os.path.exists(name_):
        os.makedirs(name_)
    if not no_describe:
      describe_data(train_path_=train_path_,
                    test_path_=test_path_,
                    base_path=name_)

    if not no_rnn_:
        train_RNN(train_path_=train_path_,
                  test_path_=test_path_,
                  base_path=name_,
                  prefix=name_ + "_",
                  search_trials=search_trials,
                  random_trials=random_trials,
                  load_emb=load_emb,
                  bidirectional=bidirectional,
                  freeze_emb=freeze_emb,
                  opt=opt,
                  Model=Model,
                  toy=toy)

    if not no_bert_:
        train_BERT(train_path_=train_path_,
                   test_path_=test_path_,
                   base_path_=name_,
                   prefix=name_,
                   args_=args_)
    if not no_baseline_:
        train_bow_xgb_classifier(train_path_=train_path_,
                                 test_path_=test_path_,
                                 base_path_=name_,
                                 num_words=num_words_,
                                 toy=toy,
                                 reduce_str=reduce_str_)


def main():
    msg = """Train RNN, BERT and baseline models on data"""
    parser = argparse.ArgumentParser(description=msg)

    parser.add_argument('train_path',
                        type=str,
                        help='path to train csv')

    parser.add_argument('test_path',
                        type=str,
                        help='path to test csv')

    parser.add_argument('name',
                        type=str,
                        help='experiment name')

    parser.add_argument("-toy",
                        "--toy",
                        action="store_true",
                        default=False,
                        help="Debug param (default=False)")

    parser.add_argument("--explicit",
                        action="store_true",
                        default=False,
                        help="Param to perform multiclass classification (default=False)")  # noqa

    parser.add_argument('--no_bert',
                        action='store_true',
                        default=False,
                        help="param to stop using bert")

    parser.add_argument('--no_describe',
                        action='store_true',
                        default=False,
                        help="param to stop performing text description")

    parser.add_argument('--no_rnn',
                        action='store_true',
                        default=False,
                        help="param to stop using bert")

    parser.add_argument('--no_baseline',
                        action='store_true',
                        default=False,
                        help="param to stop using bert")

    parser.add_argument("-nw",
                        "--num_words",
                        type=int,
                        default=1000,
                        help="(BASELINE) Number of words used for the baseline model (default=1000)")  # noqa

    parser.add_argument('--reduce_str',
                        action='store_true',
                        default=False,
                        help="(BASELINE) param to remove stopwords and apply stemming")  # noqa

    parser.add_argument("-m",
                        "--model",
                        type=str,
                        default="RNN",
                        help="(RNN) model type: 'RNN', 'GRU', 'LSTM' (default=RNN)")  # noqa

    parser.add_argument("-st",
                    "--search_trials",
                    type=int,
                    default=2,
                    help="(RNN) number of times to call the grid seach funtion(default=2)")  # noqa

    parser.add_argument("-rt",
                    "--random_trials",
                    type=int,
                    default=2,
                    help="(RNN) number of times to call the random seach funtion(default=2)")  # noqa

    parser.add_argument("-em",
                    "--embedding",
                    type=str,
                    default="None",
                    help="(RNN) pre trained word embedding (default=None)")  # noqa

    parser.add_argument("-bi",
                        "--bidirectional",
                        action="store_true",
                        default=False,
                        help="(RNN) Use bidirectional rnn (default=False)")

    parser.add_argument("-f",
                        "--freeze_emb",
                        action="store_true",
                        default=False,
                        help="(RNN) freeze embedding layer (default=False)")

    parser.add_argument("-o",
                        "--optmizer",
                        type=str,
                        default="sgd",
                        help="(RNN) torch optmizer: 'sgd', 'adam', 'adagrad' 'rmsprop' (default=sgd)")  # noqa

    parser.add_argument("--bert_model",
                        default="bert-base-uncased",
                        type=str,
                        help="(BERT) Bert pre-trained model selected in the list: bert-base-uncased, "  # noqa
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "  # noqa
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="(BERT) The maximum total input sequence length after WordPiece tokenization. \n"  # noqa
                        "Sequences longer than this will be truncated, and sequences shorter \n"  # noqa
                        "than this will be padded.")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="(BERT) Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="(BERT) Total batch size for eval.")

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="(BERT) The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="(BERT) Total number of training epochs to perform.")  # noqa

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="(BERT) Proportion of training to perform linear learning rate warmup for. "  # noqa
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="(BERT) local_rank for distributed training on gpus")  # noqa

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="(BERT) random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="(BERT) Number of updates steps to accumulate before performing a backward/update pass.")  # noqa

    parser.add_argument('--fp16',
                        action='store_true',
                        help="(BERT) Whether to use 16-bit float precision instead of 32-bit")  # noqa

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="(BERT) Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"  # noqa
                             "0 (default value): dynamic loss scaling.\n"  # noqa
                             "Positive power of 2: static loss scaling value.\n")  # noqa

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="(BERT) Set this flag if you are using an uncased model.")  # noqa

    models_and_names = {"RNN": RNN, "GRU": GRU, "LSTM": LSTM}
    embedding_and_names = {"None": None,
                           "charngram": "charngram.100d",
                           "fasttextEn": "fasttext.en.300d",
                           "fasttextSimple": "fasttext.simple.300d",
                           "glove42": "glove.42B.300d",
                           "glove84": "glove.840B.300d",
                           "gloveTwitter25": "glove.twitter.27B.25d",
                           "gloveTwitter50": "glove.twitter.27B.50d",
                           "gloveTwitter100": "glove.twitter.27B.100d",
                           "gloveTwitter200": "glove.twitter.27B.200d",
                           "glove6b_80": "glove.6B.50d",
                           "glove6b_100": "glove.6B.100d",
                           "glove6b_200": "glove.6B.200d",
                           "glove6b_300": "glove.6B.300d"}
    all_opts = ['sgd', 'adam', 'adagrad', 'rmsprop']

    args = parser.parse_args()
    user_model = args.model.upper()
    msg = "not a valid model"
    assert user_model in models_and_names, msg
    opt = args.optmizer.lower().strip()
    msg = "not a valid opt"
    assert opt in all_opts, msg

    Model = models_and_names[user_model]
    load_emb = embedding_and_names[args.embedding]
    search_trials = args.search_trials
    random_trials = args.random_trials
    bidirectional = args.bidirectional
    freeze = args.freeze_emb
    opt = args.optmizer

    if not args.explicit:
        task = "contraboolean"
    else:
        task = "explicit"

    BERTargs = {"local_rank": args.local_rank,
                "fp16": args.fp16,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                'learning_rate': args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "seed": args.seed,
                'max_seq_length': args.max_seq_length,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,  # noqa
                "loss_scale": args.loss_scale,
                "do_lower_case": args.do_lower_case,
                "toy": args.toy,
                "bert_model": args.bert_model,
                "task_name": task,
                "warmup_proportion": args.warmup_proportion}

    train(train_path_=args.train_path,
          test_path_=args.test_path,
          name_=args.name,
          args_=BERTargs,
          search_trials=search_trials,
          random_trials=random_trials,
          load_emb=load_emb,
          bidirectional=bidirectional,
          freeze_emb=freeze,
          opt=opt,
          Model=Model,
          toy=args.toy,
          no_rnn_=args.no_rnn,
          no_bert_=args.no_bert,
          no_baseline_=args.no_baseline,
          num_words_=args.num_words,
          reduce_str_=args.reduce_str,
          no_describe=args.no_describe)


if __name__ == '__main__':
    main()


