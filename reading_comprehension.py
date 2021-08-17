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
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.data.processors.squad import SquadV2Processor
from transformers.data.metrics.squad_metrics import normalize_answer, compute_exact, compute_f1, merge_eval, \
    make_eval_dict, apply_no_ans_threshold, find_all_best_thresh, squad_evaluate

from cleantext import clean

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class ReadingComprehensionDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Reading Comprehension. """

    def __init__(self, input_data, tokenizer, max_length):
        self.input_data = input_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, item):
        pair = self.input_data[item]['context'] + "\n" + self.input_data[item]['question']
        encoding = self.tokenizer(
            pair,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        inputs = {
            'qas_id': self.input_data[item]['qas_id'],
            'context_question': pair,
            'context': self.input_data[item]['context'],
            'question': self.input_data[item]['question'],
            'input_ids': encoding.input_ids.flatten(),
            'attention_mask': encoding.attention_mask.flatten()
        }
        return inputs


class ReadingComprehension:
    def __init__(self, model_name, model_type):
        self.normalizer = hazm.Normalizer()
        self.model_name = model_name
        if model_type.lower() == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
            self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
            self.config = MT5Config.from_pretrained(self.model_name)
        elif model_type.lower() == "GPT2LMHeadModel":
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.id2label = self.config.id2label
            self.label2id = self.config.label2id

    def load_dataset_test_file(self, dataset_name, dataset_file, **kwargs):
        if dataset_name.lower() == "parsinlu":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            processor = SquadV2Processor()
            examples = processor.get_dev_examples(dataset_file[:dataset_file.rfind('/')],
                                                  filename=dataset_file[dataset_file.rfind('/') + 1:])
            return examples

    def mt5_reading_comprehension_inference(self, context, question, device):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return

        new_input = []
        for q1, q2 in zip(context, question):
            new_input.append(q1 + "\n" + q2)

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

    def gpt2_reading_comprehension_inference(self, context, question, device, max_length):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()
        # Set the model in evaluation mode to deactivate the DropOut modules
        # This is IMPORTANT to have reproducible results during evaluation!
        self.model.eval()

        for q1, q2 in zip(context, question):
            input_text = q1 + " پرسش: " + q2 + " پاسخ: "
            print(f"Input text:\n{input_text}")
            encoded_input = self.tokenizer.encode(input_text, return_tensors="pt")
            encoded_input = encoded_input.to(device)
            print("encoded_input:", encoded_input)
            output = self.model.generate(encoded_input, max_length=max_length)
            print("output:", output)
            generated_text = self.tokenizer.batch_decode(output)
            print(f"Generated text:\n{generated_text}")
            if generated_text.startswith(input_text):
                print(f"Answer:\n{generated_text[len(input_text):]}")

    def mt5_evaluation(self, examples, device, max_length, batch_size=4):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return

        input_data = []
        for example in examples:
            gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]
            if not gold_answers:
                # For unanswerable questions, only correct answer is empty string
                gold_answers = [""]
            input_data.append({
                "qas_id": example.qas_id,
                "context": example.context_text,
                "question": example.question_text,
                "answers": example.answers,
                "gold_answers": gold_answers
            })

        qas_answers = {sample['qas_id']: sample['gold_answers'] for sample in input_data}
        dataset = ReadingComprehensionDataset(input_data=input_data, tokenizer=self.tokenizer, max_length=max_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print(f'#input_data:{len(input_data)}')
        print("#batch:", len(data_loader))

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        total_time = 0
        output_predictions = []
        exact_scores, f1_scores, no_answer_probs, preds = {}, {}, {}, {}
        print("Start to evaluate test data ...")
        for step, batch in enumerate(data_loader):
            b_qas_id = batch['qas_id']
            b_context_question = batch['context_question']
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
            for i in range(len(b_predictions)):
                qas_id = b_qas_id[i]
                gold_answers = qas_answers[qas_id]
                predicted_answer = b_predictions[i]
                exact_scores[qas_id] = max(compute_exact(a, predicted_answer) for a in gold_answers)
                f1_scores[qas_id] = max(compute_f1(a, predicted_answer) for a in gold_answers)

                output_predictions.append((
                    qas_id,
                    batch['context'][i],
                    batch['question'][i],
                    gold_answers,
                    predicted_answer,
                    exact_scores[qas_id],
                    f1_scores[qas_id]
                ))
                no_answer_probs[qas_id] = 0.0
                preds[qas_id] = predicted_answer

        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(input_data))

        # evaluate
        evaluation_results = squad_evaluate(examples, preds)
        print("evaluation results:\n", evaluation_results)
        return evaluation_results, output_predictions
