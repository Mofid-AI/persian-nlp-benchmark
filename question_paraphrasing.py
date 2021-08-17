import re
import gc
import os
import hazm
import time
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import MT5Config, MT5ForConditionalGeneration, MT5Tokenizer

from cleantext import clean

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class QuestionParaphrasingDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Question Paraphrasing. """

    def __init__(self, first_questions, second_questions, targets, tokenizer, max_length):
        self.first_questions = first_questions
        self.second_questions = second_questions
        self.tokenizer = tokenizer
        self.targets = targets
        self.max_length = max_length

    def __len__(self):
        return len(self.first_questions)

    def __getitem__(self, item):
        pair = self.first_questions[item] + "<sep>" + self.second_questions[item]
        encoding = self.tokenizer(
            pair,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        inputs = {
            'q1': self.first_questions[item],
            'q2': self.second_questions[item],
            'pair': pair,
            'targets': self.targets[item],
            'input_ids': encoding.input_ids.flatten(),
            'attention_mask': encoding.attention_mask.flatten()
        }
        return inputs


class QuestionParaphrasing:
    def __init__(self, model_name, model_type):
        self.normalizer = hazm.Normalizer()
        self.model_name = model_name
        if model_type.lower() == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
            self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
            self.config = MT5Config.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.id2label = self.config.id2label
            self.label2id = self.config.label2id

    def load_dataset_test_file(self, dataset_name, dataset_file, **kwargs):
        if dataset_name.lower() == "qqp":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            question_1, question_2, labels, categories = [], [], [], []
            with open(dataset_file, encoding="utf8") as infile:
                for line in infile:
                    json_line = json.loads(line.strip())
                    q1 = json_line['q1']
                    q2 = json_line['q2']
                    label = json_line['label']
                    category = json_line['category']  # qqp or natural
                    if 'category' in kwargs and kwargs['category'] != category:
                        continue
                    question_1.append(q1)
                    question_2.append(q2)
                    labels.append(label)
                    categories.append(category)
            return question_1, question_2, labels, categories

    def mt5_question_paraphrasing_inference(self, input_text_1, input_text_2, device):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return

        new_input = []
        for q1, q2 in zip(input_text_1, input_text_2):
            new_input.append(q1 + "<sep>" + q2)

        tokenized_batch = self.tokenizer(
            new_input,
            padding=True,
            return_tensors="pt"
        )

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        input_ids = tokenized_batch.input_ids.to(device)
        attention_mask = tokenized_batch.attention_mask.to(device)
        outputs = self.model.generate(input_ids=input_ids,
                                      attention_mask=attention_mask)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predictions

    def mt5_evaluation(self, input_text_1, input_text_2, input_labels, device, max_length, batch_size=4):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return
        if len(input_text_1) != len(input_text_2):
            print('length of two inputs is not equal!!')
            return
        if len(input_text_1) != len(input_labels):
            print('length of inputs and labels is not equal!!')
            return

        dataset = QuestionParaphrasingDataset(first_questions=input_text_1, second_questions=input_text_2,
                                              targets=input_labels, tokenizer=self.tokenizer, max_length=max_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print(f'#q1:{len(input_text_1)}, #q2:{len(input_text_2)}')
        print("#batch:", len(data_loader))

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        total_time = 0
        output_predictions = []
        golden_labels, predicted_labels = [], []
        print("Start to evaluate test data ...")
        for step, batch in enumerate(data_loader):
            b_input_ids = batch['input_ids']
            b_attention_mask = batch['attention_mask']

            # move tensors to GPU if CUDA is available
            b_input_ids = b_input_ids.to(device)
            b_attention_mask = b_attention_mask.to(device)

            # This will return the loss (rather than the model output) because we have provided the `labels`.
            with torch.no_grad():
                start = time.monotonic()
                b_outputs = self.model.generate(input_ids=b_input_ids, attention_mask=b_attention_mask)
                end = time.monotonic()
                total_time += end - start
                print(f'inference time for step {step}: {end - start}')

            b_targets = batch['targets']
            golden_labels.extend(b_targets)

            b_predictions = self.tokenizer.batch_decode(b_outputs, skip_special_tokens=True)
            predicted_labels.extend(b_predictions)

            for i in range(len(b_input_ids)):
                output_predictions.append((batch['q1'][i], batch['q2'][i], b_targets[i], b_predictions[i]))

        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(input_text_1))

        # evaluate
        print("Test Accuracy: {}".format(accuracy_score(golden_labels, predicted_labels)))
        print("Test Precision: {}".format(precision_score(golden_labels, predicted_labels, average="weighted")))
        print("Test Recall: {}".format(recall_score(golden_labels, predicted_labels, average="weighted")))
        print("Test F1-Score(weighted average): {}".format(
            f1_score(golden_labels, predicted_labels, average="weighted")))
        print("Test classification Report:\n{}".format(
            classification_report(golden_labels, predicted_labels, digits=10)))
        return output_predictions
