import re
import pandas as pd
from collections import Counter
import numpy as np
from functools import reduce
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
import os
import pickle
from collections import Counter

pd.options.mode.chained_assignment = None

spaces = re.compile(' +')
stopwords = nltk.corpus.stopwords.words()
stemmer = PorterStemmer()


def get_vocab_size(df, column="text"):
    def f(x, y): return x + " " + y  # noqa
    text = reduce(f, df[column].values)
    text = text.strip()
    text_c = Counter(text.split(" "))
    return len(text_c)


def get_vocab(df, column="text"):
    def f(x, y): return x + " " + y  # noqa
    text = reduce(f, df[column].values)
    text = text.strip()
    text_c = Counter(text.split(" "))
    return text_c


def stopwords_f(x): return x if x not in stopwords else ""  # noqa


def stemmer_f(x): return stemmer.stem(x)  # noqa


def normalizeString(s):
    """
    remove stop words and stemming
    """
    s = " ".join(list(map(stopwords_f, s.split())))
    s = " ".join(list(map(stemmer_f, s.split())))
    return s


def simplify_text_df(df, reduce_str):
    df["text"] = df["sentence1"] + " " + df["sentence2"]
    df = df[["text", "label"]]
    simple_pre_process_text_df(df)
    if reduce_str:
        df["text"] = df["text"].apply(normalizeString)
    return df


def remove_first_space(x):
    """
    :param x: word
    :type x: str
    :return: word withou space in front
    :rtype: str
    """
    if x[0] == " ":
        return x[1:]
    else:
        return x


def simple_pre_process_text_df(data, key='text'):
    """
    :param data: data frame with the colum 'text'
    :type data: pd.DataFrame
    :param key: colum key
    :type key: str
    """

    data[key] = data[key].apply(lambda x: x.lower())
    data[key] = data[key].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))  # noqa
    data[key] = data[key].apply(remove_first_space)  # noqa remove space in the first position
    data[key] = data[key].apply((lambda x: spaces.sub(" ", x)))  # noqa remove double spaces


def describe_data(train_path_,
                  test_path_,
                  base_path):
    df_train = pd.read_csv(train_path_)
    df_test = pd.read_csv(test_path_)
    train_size = df_train.shape[0]
    test_size = df_test.shape[0]
    df_train["text"] = df_train['sentence1'] + " " + df_train['sentence2']
    df_train["len"] = df_train['text'].apply(lambda x: len(x))
    df_test["text"] = df_test['sentence1'] + " " + df_test['sentence2']
    df_test["len"] = df_test['text'].apply(lambda x: len(x))
    df = pd.concat([df_train, df_test])
    full_vocab_size = get_vocab_size(df)
    c_train = Counter(df_train["label"].values)
    c_test = Counter(df_test["label"].values)
    train_vocab = get_vocab(df_train)
    test_vocab = get_vocab(df_test)
    train_vocab_size = len(train_vocab)
    test_vocab_size = len(test_vocab)
    intersection_size = len(
        set(train_vocab.keys()).intersection(test_vocab.keys()))
    mean_input_lenght_train = np.mean(df_train["len"].values)
    mean_input_lenght_test = np.mean(df_test["len"].values)
    mean_input_lenght = np.mean(df["len"].values)
    max_input_lenght = np.max(df["len"].values)
    std_input_lenght_train = np.std(df_train["len"].values)
    std_input_lenght_test = np.std(df_test["len"].values)
    std_input_lenght = np.std(df["len"].values)
    write_path = os.path.join(base_path, "data_description.txt")  # noqa
    with open(write_path, "w") as file:
        file.write("===Data description===\n")
        file.write("\n==Train==\n\n")
        file.write("train path =  {}\n".format(train_path_))
        file.write("train size =  {}\n".format(train_size))
        file.write("train label distribution =  {}\n".format(c_train))
        file.write("train vocab size =  {}\n".format(train_vocab_size))
        file.write("train mean input lenght =  {:.2f}\n".format(
            mean_input_lenght_train))
        file.write("train std input lenght =  {:.2f}\n".format(
            std_input_lenght_train))
        file.write("\n==Test==\n\n")
        file.write("test path =  {}\n".format(test_path_))
        file.write("test size =  {}\n".format(test_size))
        file.write("test label distribution =  {}\n".format(c_test))
        file.write("test vocab size =  {}\n".format(test_vocab_size))
        file.write("test mean input lenght =  {:.2f}\n".format(
            mean_input_lenght_test))
        file.write("test std input lenght =  {:.2f}\n".format(
            std_input_lenght_test))
        file.write("\n==General==\n\n")
        file.write("vocab size =  {}\n".format(full_vocab_size))
        file.write(
            "train/test vocab intersection size =  {}\n".format(intersection_size))  # noqa
        file.write("mean input lenght =  {:.2f}\n".format(mean_input_lenght))
        file.write("std input lenght =  {:.2f}\n".format(std_input_lenght))
        file.write("max input lenght =  {:.2f}\n".format(max_input_lenght))



def NLI2Contra(input_path,
               output_path,
               input_colum_label="gold_label",
               input_colum_sentence1="sentence1",
               input_colum_sentence2="sentence2",
               sep="\t"):
    """

    Transfor a NLI dataset into a contradiction dataset

    :param data: data frame with the colum 'text'
    :type data: pd.DataFrame
    :param key: colum key
    :type key: str
    """

    df_ = pd.read_csv(input_path, sep="\t")
    df_["old_label"] = df_[input_colum_label]
    df_["sentence1"] = df_[input_colum_sentence1]
    df_["sentence2"] = df_[input_colum_sentence2]
    df_ = df_[["sentence1", "sentence2", "old_label"]]
    df_.dropna(how='any', inplace=True)
    simple_pre_process_text_df(df_, key="sentence1")
    simple_pre_process_text_df(df_, key="sentence2")
    df_contra = df_.loc[df_["old_label"] == 'contradiction']
    df_non_contra = df_.loc[(df_["old_label"] == 'neutral') | (
        df_["old_label"] == 'entailment')]
    df_non_contra = df_non_contra.sample(frac=1).reset_index(drop=True)
    df_contra = df_contra.sample(frac=1).reset_index(drop=True)
    df_non_contra["label"] = [0] * df_non_contra.shape[0]
    df_contra["label"] = [1] * df_contra.shape[0]
    df_non_contra = df_non_contra.head(df_contra.shape[0])
    df_clean = pd.concat([df_contra, df_non_contra])
    df_clean = df_clean.sample(frac=1).reset_index(drop=True)
    df_clean.drop(["old_label"], axis=1, inplace=True)
    df_clean.to_csv(output_path, header=True, index=False)


def train_bow_xgb_classifier(train_path_,
                             test_path_,
                             base_path_,
                             num_words,
                             toy,
                             reduce_str):
    "XGBoost is an implementation of gradient boosted decision trees"
    if not os.path.exists(base_path_):
        os.makedirs(base_path_)
    df_train = pd.read_csv(train_path_)
    df_test = pd.read_csv(test_path_)
    df_train = simplify_text_df(df_train, reduce_str)
    df_test = simplify_text_df(df_test, reduce_str)
    if toy:
        df_train = df_train.head(100)
        df_test = df_test.head(10)
    vocab_size = get_vocab_size(df_train)
    if num_words > vocab_size:
        num_words = vocab_size
    train_corpus = list(df_train["text"].values)
    vectorizer = CountVectorizer(max_features=num_words)
    vectorizer.fit(train_corpus)
    X_train = vectorizer.transform(train_corpus).toarray()
    y_train = df_train["label"].values
    xgb_classifier = XGBClassifier()
    xgb_classifier.fit(X_train, y_train, verbose=True)
    test_corpus = list(df_test["text"].values)
    X_test = vectorizer.transform(test_corpus).toarray()
    y_test = df_test["label"].values
    pred = xgb_classifier.predict(X_test)
    accuracy = np.mean(pred == y_test)
    df_test = pd.read_csv(test_path_)
    if toy:
        df_test = df_test.head(10)
    df_test["pred"] = pred
    df_test_hit = df_test.loc[pred == y_test]
    df_test_miss = df_test.loc[np.logical_not(pred == y_test)]
    base_path_outputs = os.path.join(base_path_, "baseline_outputs")
    base_path_models = os.path.join(base_path_, "baseline_models")
    if not os.path.exists(base_path_outputs):
        os.makedirs(base_path_outputs)
    if not os.path.exists(base_path_models):
        os.makedirs(base_path_models)
    write_path = os.path.join(base_path_outputs, "baseline_results.txt")  # noqa
    with open(write_path, "w") as file:
        file.write("acc =  {:.3f}\n".format(accuracy))
        file.write("remove stopwords and apply stemming =  {}\n".format(reduce_str))  # noqa
        file.write("number of words =  {}\n".format(num_words))
        file.write("model details\n{}".format(str(xgb_classifier)))
    df_test_hit.to_csv(os.path.join(base_path_outputs, "hit.csv"), header=True, index=False)  # noqa
    df_test_miss.to_csv(os.path.join(base_path_outputs, "miss.csv"), header=True, index=False)  # noqa
    with open(os.path.join(base_path_models, "baseline.pickle"), 'wb') as file:
        di = {'xgb_classifier': xgb_classifier, 'vectorizer': vectorizer}
        pickle.dump(di, file)
