import pandas as pd
import unittest
import os

from clcd.text_processing.functions import simple_pre_process_text_df
from clcd.bert_train.classification import get_training_obj, train_model
from clcd.bert_train.classification import ContraProcessor, eval_model


class BERTtrainer(unittest.TestCase):

    @classmethod
    def tearDown(cls):
        if os.path.exists(cls.save_path):
            os.remove(cls.save_path)
        if os.path.exists(cls.out_path):
            os.remove(cls.out_path)

    @classmethod
    def setUp(cls):
        cls.args = {"local_rank": -1,
                    "fp16": False,
                    "train_batch_size": 32,
                    "eval_batch_size": 8,
                    'learning_rate': 5e-5,
                    "num_train_epochs": 1.0,
                    "seed": 42,
                    'max_seq_length': 128,
                    "gradient_accumulation_steps": 1,
                    "loss_scale": 0,
                    "do_lower_case": False,
                    "do_train": False,
                    "bert_model": "bert-base-uncased",
                    "task_name": None,
                    "warmup_proportion": 0.1}

        cls.args["task_name"] = "contraboolean"

        cls.processors = {"contraboolean": ContraProcessor}
        cls.num_labels_task = {"contraboolean": 2}

        cls.df_name = "boolean1_UNITTEST"
        train_path = os.path.join("text_gen_output", "boolean1_train.csv")
        test_path = os.path.join("text_gen_output", "boolean1_test.csv")
        df_train = pd.read_csv(train_path)
        simple_pre_process_text_df(df_train, key="sentence1")
        simple_pre_process_text_df(df_train, key="sentence2")
        df_dev = pd.read_csv(test_path)
        simple_pre_process_text_df(df_dev, key="sentence1")
        simple_pre_process_text_df(df_dev, key="sentence2")
        cls.df_train = df_train.head(100)
        cls.df_dev = df_dev.head(100)
        cls.save_path = os.path.join("models",
                                     "boolean1_UNITTEST_pytorch_model.bin")
        cls.out_path = os.path.join("outputs",
                                    "boolean1_UNITTEST_eval_results.txt")

    def test_model_is_saved_after_training_and_can_evaluate_test(self):  # noqa
        obj_dict = get_training_obj(args_=self.args,
                                    df_train_=self.df_train,
                                    df_dev_=self.df_dev,
                                    processors_=self.processors,
                                    num_labels_task_=self.num_labels_task)

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
                                              model_name_=self.df_name,
                                              train_examples_=train_examples,
                                              label_list_=label_list,
                                              args_=args,
                                              tokenizer_=tokenizer,
                                              logger_=logger,
                                              num_train_steps_=num_train_steps,
                                              optimizer_=optimizer,
                                              n_gpu_=n_gpu,
                                              t_total_=t_total,
                                              device_=device)

        opath = eval_model(output_model_file_=path,
                           model_name_=self.df_name,
                           eval_examples_=eval_examples,
                           args_=args,
                           logger_=logger,
                           tokenizer_=tokenizer,
                           device_=device,
                           num_labels_=num_labels,
                           label_list_=label_list,
                           loss_=loss,
                           global_step_=global_step)

        self.assertTrue(os.path.exists(self.save_path))
        self.assertTrue(os.path.exists(opath))
