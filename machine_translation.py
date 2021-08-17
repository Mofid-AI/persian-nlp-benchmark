import re
import gc
import os
import hazm
import time
import json
import sacrebleu
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


class MachineTranslationDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Machine Translation. """

    def __init__(self, original_text, translated_text, tokenizer, max_length):
        self.original_text = original_text
        self.translated_text = translated_text
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.original_text)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            self.original_text[item],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        inputs = {
            'original': self.original_text[item],
            'translated': self.translated_text[item],
            'input_ids': encoding.input_ids.flatten(),
            'attention_mask': encoding.attention_mask.flatten()
        }
        return inputs


class MachineTranslation:
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
        if dataset_name.lower() == "mizan":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            data = pd.read_csv(dataset_file, delimiter="\t", names=['original', 'translation'], header=None)
            original, translation = data['original'].values.tolist(), data['translation'].values.tolist()
            print(f'test part:\n #original: {len(original)}, #translation: {len(translation)}')
            return original, translation
        if dataset_name.lower() == "combined":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            data = pd.read_csv(dataset_file, delimiter="\t", names=['original', 'translation', 'source'], header=None)
            if 'source' in kwargs:
                data = data[data['source'] == kwargs['source']]
                data = data[['original', 'translation']]
            else:
                data = data[['original', 'translation']]
            original, translation = data['original'].values.tolist(), data['translation'].values.tolist()
            print(f'test part:\n #original: {len(original)}, #translation: {len(translation)}')
            return original, translation

    def load_dataset_file(self, dataset_name, dataset_file, **kwargs):
        if dataset_name.lower() == "quran" or dataset_name.lower() == "bible":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            original, translation = [], []
            with open(dataset_file, encoding="utf8") as infile:
                for line in infile:
                    parts = line.strip().split('\t')
                    original.append(parts[0])
                    translation.append(parts[1])
            print(f'all data:\n #original: {len(original)}, #translation: {len(translation)}')

            _, original_test, _, translation_test = train_test_split(original, translation, test_size=0.1,
                                                                     random_state=1)
            print(f'test part:\n #original: {len(original_test)}, #translation: {len(translation_test)}')
            return original, translation, original_test, translation_test

    def mt5_machine_translation_inference(self, input_text, device):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return

        tokenized_batch = self.tokenizer(
            input_text,
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

    def mt5_evaluation(self, input_text, translated_text, device, max_length, split_reference=None, batch_size=4):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return
        if len(input_text) != len(translated_text):
            print('length of inputs and its translations is not equal!!')
            return

        dataset = MachineTranslationDataset(original_text=input_text, translated_text=translated_text,
                                            tokenizer=self.tokenizer, max_length=max_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print(f'#original_text:{len(input_text)}, #translated_text:{len(translated_text)}')
        print("#batch:", len(data_loader))

        if split_reference is None:
            max_num_ref = 1
        else:
            max_num_ref = 0
            for ref in translated_text:
                max_num_ref = max(max_num_ref, len(ref.split(split_reference)))
        print("#maximum_translation_reference:", max_num_ref)

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        total_time = 0
        output_predictions = []
        golden_translations, predicted_translations = [[] for _ in range(max_num_ref)], []
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
                if split_reference is None:
                    sample_golden_translation = [batch['translated'][i]]
                else:
                    sample_golden_translation = batch['translated'][i].split(split_reference)
                sample_generated_translation = b_predictions[i]
                bleu_score = sacrebleu.corpus_bleu(sys_stream=[sample_generated_translation],
                                                   ref_streams=[[g] for g in sample_golden_translation]).score
                output_predictions.append((batch['original'][i], batch['translated'][i], b_predictions[i], bleu_score))

                for j in range(max_num_ref):
                    try:
                        golden_translations[j].append(sample_golden_translation[j])
                    except:
                        golden_translations[j].append('')
                predicted_translations.append(sample_generated_translation)

        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(input_text))

        # evaluate
        print("BLEU Score: {}".format(sacrebleu.corpus_bleu(
            sys_stream=predicted_translations, ref_streams=golden_translations).score))
        return output_predictions
