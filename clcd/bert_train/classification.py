import numpy as np
import pandas as pd
import torch
import logging
import random
import os
from tqdm import tqdm, trange


from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
# from pytorch_pretrained_bert.modeling import BertConfig


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text
            of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string.
            The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class ContraProcessor(DataProcessor):
    """Processor for the contradiction data set."""

    def get_train_examples(self, data_df):
        """See base class."""
        return self._create_examples(data_df, "train")

    def get_dev_examples(self, data_df):
        """See base class."""
        return self._create_examples(data_df, "dev")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, data_df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(data_df.shape[0]):
            data_df.iloc[i]
            guid = str(i) + "_" + set_type
            text_a = data_df.iloc[i].sentence1
            text_b = data_df.iloc[i].sentence2
            label = data_df.iloc[i].label
            ex = InputExample(guid=guid,
                              text_a=text_a,
                              text_b=text_b,
                              label=label)
            examples.append(ex)
        return examples


class ExplicitProcessor(DataProcessor):
    """Processor for the explicit data set."""

    def get_train_examples(self, data_df):
        """See base class."""
        return self._create_examples(data_df, "train")

    def get_dev_examples(self, data_df):
        """See base class."""
        return self._create_examples(data_df, "dev")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3, 4, 5]

    def _create_examples(self, data_df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i in range(data_df.shape[0]):
            data_df.iloc[i]
            guid = str(i) + "_" + set_type
            text_a = data_df.iloc[i].sentence1
            text_b = data_df.iloc[i].sentence2
            label = data_df.iloc[i].label
            ex = InputExample(guid=guid,
                              text_a=text_a,
                              text_b=text_b,
                              label=label)
            examples.append(ex)
        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time.
    # This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def remove_helpers(str_tokens):
    helpers = [101, 102, 0]
    return [i for i in str_tokens if i not in helpers]


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 logger):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        # tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP] # noqa
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence.
        # The embedding vectors for `type=0` and
        # `type=1` were learned during
        # pre-training and are added to the wordpiece
        # embedding vector (and position vector).
        # This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences,
        # but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks,
        # the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector".
        # Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def get_training_obj(args_, df_train_, df_dev_, processors_, num_labels_task_):

    time_stamp = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'

    logging.basicConfig(format=time_stamp,
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)
    device = "cpu"
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(  # noqa
        device, n_gpu, bool(args_["local_rank"] != -1), args_["fp16"]))

    args_["train_batch_size"] = int(args_["train_batch_size"] / args_["gradient_accumulation_steps"])  # noqa

    random.seed(args_["seed"])
    np.random.seed(args_["seed"])
    torch.manual_seed(args_["seed"])
    task_name = args_["task_name"].lower()

    processor = processors_[task_name]()
    num_labels = num_labels_task_[task_name]
    label_list = processor.get_labels()  # noqa
    tokenizer = BertTokenizer.from_pretrained(args_["bert_model"], do_lower_case=args_["do_lower_case"])  # noqa

    train_examples_ = processor.get_train_examples(df_train_)
    eval_examples_ = processor.get_dev_examples(df_dev_)

    num_train_steps = int(len(train_examples_) / args_["train_batch_size"] /
                          args_["gradient_accumulation_steps"] * args_["num_train_epochs"])  # noqa

    model_ = BertForSequenceClassification.from_pretrained(args_["bert_model"],  # noqa
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /  # noqa
                                                          'distributed_{}'.format(  # noqa
                                                              args_["local_rank"]),  # noqa
                                                          num_labels=num_labels)  # noqa

    # no pretrained BERT

    # config = BertConfig(vocab_size_or_config_json_file=32000,
    #                     hidden_size=768,
    #                     num_hidden_layers=12,
    #                     num_attention_heads=12,
    #                     intermediate_size=3072)

    # model = BertForSequenceClassification(config, num_labels=num_labels)

    model_.to(device)

    param_optimizer = list(model_.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps

    if args_["fp16"]:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")  # noqa

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args_["learning_rate"],
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args_["loss_scale"] == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=args_["loss_scale"])

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args_["learning_rate"],
                             warmup=args_["warmup_proportion"],
                             t_total=t_total)
    obj_dict = {"args": args_,
                "model": model_,
                "train_examples": train_examples_,
                "eval_examples": eval_examples_,
                "optimizer": optimizer,
                "tokenizer": tokenizer,
                "logger": logger,
                "num_train_steps": num_train_steps,
                't_total': t_total,
                "device": device,
                "n_gpu": n_gpu,
                "num_labels": num_labels,
                "label_list": label_list}
    return obj_dict


def train_model(model_,
                model_name_,
                train_examples_,
                label_list_,
                args_,
                tokenizer_,
                logger_,
                num_train_steps_,
                optimizer_,
                n_gpu_,
                t_total_,
                device_,
                base_path_model="models"):

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    train_features = convert_examples_to_features(train_examples_,
                                                  label_list_,
                                                  args_["max_seq_length"],
                                                  tokenizer_,
                                                  logger_)
    logger_.info("***** Running training *****")
    logger_.info("  Num examples = %d", len(train_examples_))
    logger_.info("  Batch size = %d", args_["train_batch_size"])
    logger_.info("  Num steps = %d", num_train_steps_)
    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args_["local_rank"] == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args_["train_batch_size"])  # noqa

    model_.train()

    for _ in trange(int(args_["num_train_epochs"]), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device_) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model_(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu_ > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args_["gradient_accumulation_steps"] > 1:
                loss = loss / args_["gradient_accumulation_steps"]

            if args_["fp16"]:
                optimizer_.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args_["gradient_accumulation_steps"] == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args_["learning_rate"] * \
                    warmup_linear(global_step / t_total_,
                                  args_["warmup_proportion"])
                for param_group in optimizer_.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer_.step()
                optimizer_.zero_grad()
                global_step += 1

    # Save a trained model
    if not os.path.exists(base_path_model):
        os.mkdir(base_path_model)
    model_to_save = model_.module if hasattr(model_, 'module') else model_
    output_model_file_ = os.path.join(base_path_model, model_name_ + "_pytorch_model.bin")  # noqa
    torch.save(model_to_save.state_dict(), output_model_file_)
    loss = tr_loss / nb_tr_steps
    return output_model_file_, loss, global_step


def eval_model(output_model_file_,
               model_name_,
               eval_examples_,
               args_,
               logger_,
               tokenizer_,
               device_,
               num_labels_,
               label_list_,
               loss_,
               global_step_,
               base_path_output="outputs"):

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(output_model_file_)
    model = BertForSequenceClassification.from_pretrained(args_["bert_model"],
                                                          state_dict=model_state_dict,  # noqa
                                                          num_labels=num_labels_)  # noqa
    model.to(device_)

    # load eval
    eval_features = convert_examples_to_features(eval_examples_,
                                                 label_list_,
                                                 args_["max_seq_length"],
                                                 tokenizer_,
                                                 logger_)

    logger_.info("***** Running evaluation *****")
    logger_.info("  Num examples = %d", len(eval_examples_))
    logger_.info("  Batch size = %d", args_["eval_batch_size"])
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args_["eval_batch_size"])
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # eval loop
    eval_t = []
    eval_p = []
    eval_l = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                              desc="Evaluating"):  # noqa
        input_str = input_ids.detach().cpu().numpy()
        for arr in input_str:
            tok = remove_helpers(arr)
            tok = [tokenizer_.ids_to_tokens[i] for i in tok]
            eval_t.append(tok)
        input_ids = input_ids.to(device_)
        input_mask = input_mask.to(device_)
        segment_ids = segment_ids.to(device_)
        label_ids = label_ids.to(device_)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids,
                                  input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)
        eval_l.append(label_ids)
        eval_p.append(np.argmax(logits, axis=1))

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'global_step': global_step_,
              'loss': loss_}

    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    output_eval_file = os.path.join(base_path_output,
                                    model_name_ + "_eval_results.txt")
    all_text = [" ".join(i) for i in eval_t]
    pred = np.concatenate(eval_p)
    label = np.concatenate(eval_l)
    df = pd.DataFrame({"text": all_text,
                       "pred": pred,
                       "label": label})
    df_hit = df.loc[pred == label]
    df_miss = df.loc[pred != label]
    out_path_hit = os.path.join(base_path_output,
                                "hit.txt")
    out_path_miss = os.path.join(base_path_output,
                                 "miss.txt")
    df_hit.to_csv(out_path_hit, index=False)
    df_miss.to_csv(out_path_miss, index=False)

    with open(output_eval_file, "w") as writer:
        logger_.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger_.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return output_eval_file
