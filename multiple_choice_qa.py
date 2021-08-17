import re
import gc
import os
import hazm
import time
import json
import collections
import numpy as np
import pandas as pd
import editdistance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModelForMultipleChoice
from transformers import MT5Config, MT5ForConditionalGeneration, MT5Tokenizer
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

from cleantext import clean

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class MultipleChoiceQADataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Multiple Choice Question Answering. """

    def __init__(self, questions, candidates, choices, answers, tokenizer, max_length, model_type):
        self.questions = questions
        self.candidates = candidates
        self.choices = choices
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        if self.model_type == "mt5":
            input_text = self.questions[item] + ' <sep> ' + ' <sep> '.join(self.candidates[item])
            encoding = self.tokenizer(
                input_text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors="pt"
            )
            inputs = {
                'item': str(item),
                'question': self.questions[item],
                'candidates': ' <sep> '.join(self.candidates[item]),
                'input_text': input_text,
                'choice': self.choices[item],
                'answer': self.answers[item],
                'input_ids': encoding.input_ids.flatten(),
                'attention_mask': encoding.attention_mask.flatten()
            }
            return inputs
        else:
            choices_input_ids, choices_attention_masks, choices_token_type_ids = [], [], []
            for c in self.candidates[item]:
                text_a = ""  # empty context
                text_b = self.questions[item] + " " + c
                inputs = self.tokenizer(
                    text_a,
                    text_b,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_overflowing_tokens=True
                )
                choices_input_ids.append(inputs.input_ids[0])
                choices_attention_masks.append(inputs.attention_mask[0])
                choices_token_type_ids.append(inputs.token_type_ids[0])

            inputs = {
                'item': str(item),
                'question': self.questions[item],
                'candidates': ' <sep> '.join(self.candidates[item]),
                'choice': int(self.choices[item]) - 1,
                'answer': self.answers[item],
                'input_ids': torch.LongTensor(choices_input_ids),
                'attention_mask': torch.LongTensor(choices_attention_masks),
                'token_type_ids': torch.LongTensor(choices_token_type_ids)
            }
            return inputs


class MultipleChoiceQA:
    def __init__(self, model_name, model_type):
        self.normalizer = hazm.Normalizer()
        self.model_name = model_name
        if model_type.lower() == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
            self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
            self.config = MT5Config.from_pretrained(self.model_name)
        elif model_type.lower() in ["mbert", "parsbert", "wikibert"]:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.model = AutoModelForMultipleChoice.from_pretrained(self.model_name, config=self.config)
            self.model_type = model_type.lower()
        else:
            print(f'model_type not supported!')
            return

    def load_dataset_test_file(self, dataset_name, dataset_file, **kwargs):
        if dataset_name.lower() in ["parsinlu", "parsinlu-literature", "parsinlu-math_and_logic",
                                    "parsinlu-common_knowledge"]:
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            questions, candidates, choices, answers = [], [], [], []
            with open(dataset_file, encoding="utf8") as infile:
                for line in infile:
                    json_line = json.loads(line.strip())
                    question = json_line['question']
                    candidate_answers = json_line['candidates']
                    choice = json_line['answer']
                    answer = candidate_answers[int(json_line['answer']) - 1]

                    questions.append(question)
                    candidates.append(candidate_answers)
                    choices.append(choice)
                    answers.append(answer)
            return questions, candidates, choices, answers

    def multiple_choice_qa_inference(self, questions, candidates, device, max_length=512):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return

        input_ids, attention_masks, token_type_ids = [], [], []
        for q, cs in zip(questions, candidates):
            choices_input_ids, choices_attention_masks, choices_token_type_ids = [], [], []
            for c in cs:
                text_a = ""  # empty context
                text_b = q + " " + c
                inputs = self.tokenizer(
                    text_a,
                    text_b,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_overflowing_tokens=True,
                )
                choices_input_ids.append(inputs.input_ids[0])
                choices_attention_masks.append(inputs.attention_mask[0])
                choices_token_type_ids.append(inputs.token_type_ids[0])
            input_ids.append(choices_input_ids)
            attention_masks.append(choices_attention_masks)
            token_type_ids.append(choices_token_type_ids)

        input_ids = torch.LongTensor(input_ids).to(device)
        attention_masks = torch.LongTensor(attention_masks).to(device)
        token_type_ids = torch.LongTensor(token_type_ids).to(device)

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()
        self.model.eval()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
        predictions = torch.argmax(outputs.logits, dim=1)
        return [(questions[i], candidates[i], candidates[i][p.item()]) for i, p in enumerate(predictions)]

    def mt5_multiple_choice_qa_inference(self, questions, candidates, device):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return

        new_input = []
        for q, cs in zip(questions, candidates):
            new_input.append(q + ' <sep> ' + ' <sep> '.join(cs))

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
        self.model.eval()

        input_ids = tokenized_batch.input_ids.to(device)
        attention_mask = tokenized_batch.attention_mask.to(device)

        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [(questions[i], candidates[i], p) for i, p in enumerate(predictions)]

    def evaluation(self, questions, candidates, choices, answers, device, max_length, batch_size=4):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return
        if len(questions) != len(candidates):
            print('length of two inputs is not equal!!')
            return
        if len(choices) != len(answers):
            print('length of choices and answers is not equal!!')
            return
        if len(questions) != len(answers):
            print('length of inputs and answers is not equal!!')
            return

        dataset = MultipleChoiceQADataset(questions=questions, candidates=candidates, choices=choices, answers=answers,
                                          tokenizer=self.tokenizer, max_length=max_length, model_type=self.model_type)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print(f'#question:{len(questions)}, #candidates:{len(candidates)}, #answer:{len(answers)}')
        print("#batch:", len(data_loader))

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()
        self.model.eval()

        total_loss, total_time = 0, 0
        output_predictions = []
        golden_choices, predicted_choices = [], []
        print("Start to evaluate test data ...")
        for step, batch in enumerate(data_loader):
            b_input_ids = batch['input_ids']
            b_attention_mask = batch['attention_mask']
            b_token_type_ids = batch['token_type_ids']
            b_choices = batch['choice']

            # move tensors to GPU if CUDA is available
            b_input_ids = b_input_ids.to(device)
            b_attention_mask = b_attention_mask.to(device)
            b_token_type_ids = b_token_type_ids.to(device)
            b_choices = b_choices.to(device)

            # This will return the loss (rather than the model output) because we have provided the `labels`.
            with torch.no_grad():
                start = time.monotonic()
                b_outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask,
                                       token_type_ids=b_token_type_ids, labels=b_choices)
                end = time.monotonic()
                total_time += end - start
                print(f'inference time for step {step}: {end - start}')
            # get the loss
            total_loss += b_outputs.loss.item()

            golden_choices.extend(b_choices.cpu().detach().numpy().tolist())
            b_predictions = torch.argmax(b_outputs.logits, dim=1)
            b_predictions = b_predictions.cpu().detach().numpy().tolist()
            predicted_choices.extend(b_predictions)

            for i in range(len(b_input_ids)):
                output_predictions.append((
                    batch['question'][i],
                    batch['candidates'][i].split(' <sep> '),
                    batch['choice'][i].item(),
                    batch['answer'][i],
                    b_predictions[i],
                    batch['candidates'][i].split(' <sep> ')[b_predictions[i]]
                ))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(data_loader)
        print("average loss:", avg_train_loss)

        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(questions))

        # evaluate
        print("Test Accuracy: {}".format(accuracy_score(golden_choices, predicted_choices)))
        print("Test Precision: {}".format(precision_score(golden_choices, predicted_choices, average="weighted")))
        print("Test Recall: {}".format(recall_score(golden_choices, predicted_choices, average="weighted")))
        print("Test F1-Score(weighted average): {}".format(
            f1_score(golden_choices, predicted_choices, average="weighted")))
        print("Test classification Report:\n{}".format(
            classification_report(golden_choices, predicted_choices, digits=10)))
        return output_predictions

    def mt5_evaluation(self, questions, candidates, choices, answers, device, max_length, batch_size=4):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return
        if len(questions) != len(candidates):
            print('length of two inputs is not equal!!')
            return
        if len(choices) != len(answers):
            print('length of choices and answers is not equal!!')
            return
        if len(questions) != len(answers):
            print('length of inputs and answers is not equal!!')
            return

        dataset = MultipleChoiceQADataset(questions=questions, candidates=candidates, choices=choices, answers=answers,
                                          tokenizer=self.tokenizer, max_length=max_length, model_type="mt5")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print(f'#question:{len(questions)}, #candidates:{len(candidates)}, #answer:{len(answers)}')
        print("#batch:", len(data_loader))

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()
        self.model.eval()

        total_time = 0
        output_predictions = []
        golden_choices, predicted_choices, exact_score_list, f1_score_list = [], [], [], []
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

            b_predictions = self.tokenizer.batch_decode(b_outputs, skip_special_tokens=True)

            for i in range(len(b_input_ids)):
                if b_predictions[i] in batch['candidates'][i].split(' <sep> '):
                    predicted_choice = str(batch['candidates'][i].split(' <sep> ').index(b_predictions[i]) + 1)
                else:
                    normalized_edit_distance_list = [
                        editdistance.distance(ca, b_predictions[i]) / max(len(ca), len(b_predictions[i])) for ca in
                        batch['candidates'][i].split(' <sep> ')
                    ]
                    predicted_choice = str(normalized_edit_distance_list.index(min(normalized_edit_distance_list)) + 1)

                golden_choices.append(batch['choice'][i])
                predicted_choices.append(predicted_choice)

                exact_score_list.append(compute_exact(batch['answer'][i], b_predictions[i]))
                f1_score_list.append(compute_f1(batch['answer'][i], b_predictions[i]))

                output_predictions.append((
                    batch['question'][i],
                    batch['candidates'][i].split(' <sep> '),
                    batch['choice'][i],
                    batch['answer'][i],
                    predicted_choice,
                    b_predictions[i],
                    exact_score_list[-1],
                    f1_score_list[-1]
                ))

        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(questions))

        # evaluate
        print("Test Accuracy: {}".format(accuracy_score(golden_choices, predicted_choices)))
        print("Test Precision: {}".format(precision_score(golden_choices, predicted_choices, average="weighted")))
        print("Test Recall: {}".format(recall_score(golden_choices, predicted_choices, average="weighted")))
        print("Test F1-Score(weighted average): {}".format(
            f1_score(golden_choices, predicted_choices, average="weighted")))
        print("Test classification Report:\n{}".format(
            classification_report(golden_choices, predicted_choices, digits=10)))

        total = len(exact_score_list)
        evaluation_results = collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_score_list) / total),
                ("f1", 100.0 * sum(f1_score_list) / total),
                ("total", total),
            ]
        )
        print("evaluation results:\n", evaluation_results)

        return output_predictions
