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

from sentence_transformers import models, SentenceTransformer, util

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class TextualEntailmentDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Textual Entailment. """

    def __init__(self, premises_hypotheses, targets, label_list, tokenizer, model_type, max_length):
        self.premises_hypotheses = premises_hypotheses
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.targets = targets
        self.max_length = max_length
        self.label2index = {label: i for i, label in enumerate(label_list)} if isinstance(label_list, list) else {}
        self.index2label = {i: label for label, i in self.label2index.items()}

    def __len__(self):
        return len(self.premises_hypotheses)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            self.premises_hypotheses[item],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        if self.model_type == "mt5":
            inputs = {
                'premise': self.premises_hypotheses[item].split('<sep>')[0],
                'hypothesis': self.premises_hypotheses[item].split('<sep>')[1],
                'pair': self.premises_hypotheses[item],
                'targets': torch.tensor(self.label2index[self.targets[item]], dtype=torch.long),
                'original_targets': self.targets[item],
                'input_ids': encoding.input_ids.flatten(),
                'attention_mask': encoding.attention_mask.flatten()
            }
        else:
            inputs = {
                'premise': self.premises_hypotheses[item].split('<sep>')[0],
                'hypothesis': self.premises_hypotheses[item].split('<sep>')[1],
                'pair': self.premises_hypotheses[item],
                'targets': torch.tensor(self.label2index[self.targets[item]], dtype=torch.long),
                'original_targets': self.targets[item],
                'input_ids': encoding.input_ids.flatten(),
                'attention_mask': encoding.attention_mask.flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten()
            }
        return inputs


class TextualEntailmentDataset2(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Textual Entailment. """

    def __init__(self, premises, hypotheses, targets, label_list, tokenizer, model_type, max_length):
        self.premises = premises
        self.hypotheses = hypotheses
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.targets = targets
        self.max_length = max_length
        self.label2index = {label: i for i, label in enumerate(label_list)} if isinstance(label_list, list) else {}
        self.index2label = {i: label for label, i in self.label2index.items()}

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            [(self.premises[item], self.hypotheses[item])],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        if self.model_type == "mt5":
            inputs = {
                'premise': self.premises[item],
                'hypothesis': self.hypotheses[item],
                'targets': torch.tensor(self.label2index[self.targets[item]], dtype=torch.long),
                'original_targets': self.targets[item],
                'input_ids': encoding.input_ids.flatten(),
                'attention_mask': encoding.attention_mask.flatten()
            }
        else:
            inputs = {
                'premise': self.premises[item],
                'hypothesis': self.hypotheses[item],
                'targets': torch.tensor(self.label2index[self.targets[item]], dtype=torch.long),
                'original_targets': self.targets[item],
                'input_ids': encoding.input_ids.flatten(),
                'attention_mask': encoding.attention_mask.flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten()
            }
        return inputs


class TextualEntailmentSimilarityDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Textual Entailment. """

    def __init__(self, premises, hypotheses, targets):
        self.premises = premises
        self.hypotheses = hypotheses
        self.targets = targets

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, item):
        inputs = {
            'item': item,
            'premise': self.premises[item],
            'hypothesis': self.hypotheses[item],
            'targets': self.targets[item]
        }
        return inputs


class TextualEntailment:
    def __init__(self, model_name, model_type, label_list):
        self.normalizer = hazm.Normalizer()
        self.model_name = model_name
        self.model_type = model_type.lower()
        self.label_list = label_list
        if self.model_type == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
            self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
            self.config = MT5Config.from_pretrained(self.model_name)
        elif self.model_type == "sentence-transformer":
            word_embedding_model = models.Transformer(model_name)
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False
            )
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.id2label = self.config.id2label
            self.label2id = self.config.label2id

    def load_dataset_test_file(self, dataset_name, dataset_file, **kwargs):
        if dataset_name.lower() in ["parsinlu-natural", "parsinlu-mnli", "parsinlu-farstail"]:
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            data = pd.read_csv(dataset_file, delimiter="\t", names=['premise_hypothesis', 'label'], header=None)

            # cleaning labels
            valid_labels = ['e', 'n', 'c']
            data['label'] = data['label'].apply(lambda r: r if r in valid_labels else None)
            data = data.dropna(subset=['label'])
            data = data.reset_index(drop=True)

            if 'label_map' in kwargs:
                data['label'] = data['label'].apply(lambda l: kwargs['label_map'][l])

            premise_hypothesis, labels = data['premise_hypothesis'].values.tolist(), data['label'].values.tolist()
            print(f'test part:\n #premise_hypothesis: {len(premise_hypothesis)}, #label: {len(labels)}')
            return premise_hypothesis, labels
        if dataset_name.lower() == "farstail":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            data = pd.read_csv(dataset_file, sep='\t')

            if 'label_map' in kwargs:
                data['label'] = data['label'].apply(lambda l: kwargs['label_map'][l])

            premises = data['premise'].values.tolist()
            hypotheses = data['hypothesis'].values.tolist()
            labels = data['label'].values.tolist()
            print(f'test part:\n #premise: {len(premises)}, #hypothesis: {len(hypotheses)}, #label: {len(labels)}')
            return premises, hypotheses, labels

    def textual_entailment_inference(self, premises, hypotheses, device, max_length):
        if not self.model or not self.tokenizer or not self.id2label:
            print('Something wrong has been happened!')
            return

        new_input = []
        for p, h in zip(premises, hypotheses):
            new_input.append((p, h))

        tokenized_batch = self.tokenizer(
            new_input,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()
        self.model.eval()

        tokenized_batch = tokenized_batch.to(device)
        outputs = self.model(**tokenized_batch)
        pt_predictions = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1)

        output_predictions = []
        for i, premise in enumerate(premises):
            output_predictions.append(
                (premise, hypotheses[i], pt_predictions[i].item(), self.label_list[pt_predictions[i].item()])
            )
        return output_predictions

    def mt5_textual_entailment_inference(self, premises, hypotheses, device):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return

        new_input = []
        for p, h in zip(premises, hypotheses):
            new_input.append(p + "<sep>" + h)

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
        outputs = self.model.generate(input_ids=input_ids,
                                      attention_mask=attention_mask)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predictions

    def sentence_transformer_textual_entailment_inference(self, premises, hypotheses, device, label_map=None):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return
        if label_map is None:
            label_map = {"0<=score<0.4": "contradiction", "0.4<=score<=0.6": "neutral", "0.6<score<=1": "entailment"}
            print(f"Setting default value for label map: {label_map}")
        elif not all('score' in cond for cond in label_map.keys()):
            print(f"All the key of label_map must contain a condition on `score` variable.\n"
                  f"For example: label_map ={label_map}")
            return

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()
        self.model.eval()

        premises_embeddings = self.model.encode(premises, convert_to_tensor=True, show_progress_bar=True)
        hypotheses_embeddings = self.model.encode(hypotheses, convert_to_tensor=True, show_progress_bar=True)

        # Compute the pair-wise cosine similarities
        similarity_scores, predicted_labels = [], []
        for i in range(len(premises)):
            cosine_score = \
                util.pytorch_cos_sim(premises_embeddings[i], hypotheses_embeddings[i]).cpu().detach().numpy()[0][0]
            similarity_scores.append(cosine_score)
            predicted_label = ''
            for exp in label_map:
                if eval(exp.replace('score', str(cosine_score))):
                    predicted_label = label_map[exp]
                    break
            predicted_labels.append(predicted_label)

        output_predictions = []
        for i, premise in enumerate(premises):
            output_predictions.append(
                (premise, hypotheses[i], similarity_scores[i], predicted_labels[i])
            )
        return output_predictions

    def evaluation(self, premise_hypothesis, labels, device, max_length, batch_size=4):
        if not self.model or not self.tokenizer or not self.id2label:
            print('Something wrong has been happened!')
            return
        label_count = {label: labels.count(label) for label in labels}
        print("label_count:", label_count)
        dataset = TextualEntailmentDataset(premises_hypotheses=premise_hypothesis, targets=labels,
                                           label_list=self.label_list, tokenizer=self.tokenizer,
                                           model_type=self.model_type, max_length=max_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print(f'#premise_hypothesis:{len(premise_hypothesis)}, #labels:{len(labels)}')
        print("#batch:", len(data_loader))

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()
        self.model.eval()

        total_loss, total_time = 0, 0
        output_predictions = []
        golden_labels, predicted_labels = [], []
        print("Start to evaluate test data ...")
        for step, batch in enumerate(data_loader):
            b_premises = batch['premise']
            b_input_ids = batch['input_ids']
            b_attention_mask = batch['attention_mask']
            b_token_type_ids = batch['token_type_ids']
            b_targets = batch['targets']

            # move tensors to GPU if CUDA is available
            b_input_ids = b_input_ids.to(device)
            b_attention_mask = b_attention_mask.to(device)
            b_token_type_ids = b_token_type_ids.to(device)
            b_targets = b_targets.to(device)

            # This will return the loss (rather than the model output) because we have provided the `labels`.
            with torch.no_grad():
                start = time.monotonic()
                b_outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask,
                                       token_type_ids=b_token_type_ids, labels=b_targets)
                end = time.monotonic()
                total_time += end - start
                print(f'inference time for step {step}: {end - start}')
            # get the loss
            total_loss += b_outputs.loss.item()

            b_original_targets = batch['original_targets']
            golden_labels.extend(b_original_targets)

            b_predictions = torch.argmax(F.softmax(b_outputs.logits, dim=1), dim=1)
            b_predictions = b_predictions.cpu().detach().numpy().tolist()
            b_predictions = [dataset.index2label[label] for label in b_predictions]
            predicted_labels.extend(b_predictions)

            for i, premise in enumerate(b_premises):
                output_predictions.append((
                    premise,
                    batch['hypothesis'][i],
                    b_original_targets[i],
                    b_predictions[i]
                ))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(data_loader)
        print("average loss:", avg_train_loss)
        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(premise_hypothesis))

        # evaluate
        print("Test Accuracy: {}".format(accuracy_score(golden_labels, predicted_labels)))
        print("Test Precision: {}".format(precision_score(golden_labels, predicted_labels, average="weighted")))
        print("Test Recall: {}".format(recall_score(golden_labels, predicted_labels, average="weighted")))
        print("Test F1-Score(weighted average): {}".format(
            f1_score(golden_labels, predicted_labels, average="weighted")))
        print("Test classification Report:\n{}".format(classification_report(
            golden_labels, predicted_labels, digits=10)))
        return output_predictions

    def evaluation_2(self, premises, hypotheses, labels, device, max_length, batch_size=4):
        if not self.model or not self.tokenizer or not self.id2label:
            print('Something wrong has been happened!')
            return
        label_count = {label: labels.count(label) for label in labels}
        print("label_count:", label_count)
        dataset = TextualEntailmentDataset2(premises=premises, hypotheses=hypotheses, targets=labels,
                                            label_list=self.label_list, tokenizer=self.tokenizer,
                                            model_type=self.model_type, max_length=max_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print(f'#premises:{len(premises)}, #hypotheses:{len(hypotheses)}, #labels:{len(labels)}')
        print("#batch:", len(data_loader))

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()
        self.model.eval()

        total_loss, total_time = 0, 0
        output_predictions = []
        golden_labels, predicted_labels = [], []
        print("Start to evaluate test data ...")
        for step, batch in enumerate(data_loader):
            b_premises = batch['premise']
            b_input_ids = batch['input_ids']
            b_attention_mask = batch['attention_mask']
            b_token_type_ids = batch['token_type_ids']
            b_targets = batch['targets']

            # move tensors to GPU if CUDA is available
            b_input_ids = b_input_ids.to(device)
            b_attention_mask = b_attention_mask.to(device)
            b_token_type_ids = b_token_type_ids.to(device)
            b_targets = b_targets.to(device)

            # This will return the loss (rather than the model output) because we have provided the `labels`.
            with torch.no_grad():
                start = time.monotonic()
                b_outputs = self.model(input_ids=b_input_ids, attention_mask=b_attention_mask,
                                       token_type_ids=b_token_type_ids, labels=b_targets)
                end = time.monotonic()
                total_time += end - start
                print(f'inference time for step {step}: {end - start}')
            # get the loss
            total_loss += b_outputs.loss.item()

            b_original_targets = batch['original_targets']
            golden_labels.extend(b_original_targets)

            b_predictions = torch.argmax(F.softmax(b_outputs.logits, dim=1), dim=1)
            b_predictions = b_predictions.cpu().detach().numpy().tolist()
            b_predictions = [dataset.index2label[label] for label in b_predictions]
            predicted_labels.extend(b_predictions)

            for i, premise in enumerate(b_premises):
                output_predictions.append((
                    premise,
                    batch['hypothesis'][i],
                    b_original_targets[i],
                    b_predictions[i]
                ))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(data_loader)
        print("average loss:", avg_train_loss)
        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(premises))

        # evaluate
        print("Test Accuracy: {}".format(accuracy_score(golden_labels, predicted_labels)))
        print("Test Precision: {}".format(precision_score(golden_labels, predicted_labels, average="weighted")))
        print("Test Recall: {}".format(recall_score(golden_labels, predicted_labels, average="weighted")))
        print("Test F1-Score(weighted average): {}".format(
            f1_score(golden_labels, predicted_labels, average="weighted")))
        print("Test classification Report:\n{}".format(classification_report(
            golden_labels, predicted_labels, digits=10)))
        return output_predictions

    def mt5_evaluation(self, premise_hypothesis, labels, device, max_length, batch_size=4):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return
        if len(premise_hypothesis) != len(labels):
            print('length of inputs and labels is not equal!!')
            return

        dataset = TextualEntailmentDataset(premises_hypotheses=premise_hypothesis, targets=labels,
                                           label_list=self.label_list, tokenizer=self.tokenizer,
                                           model_type=self.model_type, max_length=max_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print(f'#premise_hypothesis:{len(premise_hypothesis)}, #labels:{len(labels)}')
        print("#batch:", len(data_loader))

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()
        self.model.eval()

        total_time = 0
        output_predictions = []
        golden_labels, predicted_labels = [], []
        print("Start to evaluate test data ...")
        for step, batch in enumerate(data_loader):
            b_premises = batch['premise']
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

            b_original_targets = batch['original_targets']
            golden_labels.extend(b_original_targets)

            b_predictions = self.tokenizer.batch_decode(b_outputs, skip_special_tokens=True)
            predicted_labels.extend(b_predictions)

            for i, premise in enumerate(b_premises):
                output_predictions.append((
                    premise,
                    batch['hypothesis'][i],
                    b_original_targets[i],
                    b_predictions[i]
                ))

        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(premise_hypothesis))

        # evaluate
        print("Test Accuracy: {}".format(accuracy_score(golden_labels, predicted_labels)))
        print("Test Precision: {}".format(precision_score(golden_labels, predicted_labels, average="weighted")))
        print("Test Recall: {}".format(recall_score(golden_labels, predicted_labels, average="weighted")))
        print("Test F1-Score(weighted average): {}".format(
            f1_score(golden_labels, predicted_labels, average="weighted")))
        print("Test classification Report:\n{}".format(
            classification_report(golden_labels, predicted_labels, digits=10)))
        return output_predictions

    def evaluation_pair_similarity(self, premises, hypotheses, labels, device, label_map=None, batch_size=4):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return
        if label_map is None:
            label_map = {"0<=score<0.4": "contradiction", "0.4<=score<=0.6": "neutral",
                         "0.6<score<=1": "entailment"}
            print(f"Setting default value for label map: {label_map}")
        elif not all('score' in cond for cond in label_map.keys()):
            print(f"All the key of label_map must contain a condition on `score` variable.\n"
                  f"For example: label_map ={label_map}")
            return

        label_count = {label: labels.count(label) for label in labels}
        print("label_count:", label_count)
        print(f'#premises:{len(premises)}, #hypotheses:{len(hypotheses)}, #labels:{len(labels)}')

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()
        self.model.eval()

        total_time = 0
        print("Start to evaluate test data ...")

        # Compute the sentence embeddings
        start = time.monotonic()
        premises_embeddings = self.model.encode(premises, convert_to_tensor=True, show_progress_bar=True,
                                                batch_size=batch_size)
        hypotheses_embeddings = self.model.encode(hypotheses, convert_to_tensor=True, show_progress_bar=True,
                                                  batch_size=batch_size)
        end = time.monotonic()
        print(f'time for computing embeddings of premises and hypotheses: {end - start}')
        total_time += end - start

        dataset = TextualEntailmentSimilarityDataset(premises=premises_embeddings, hypotheses=hypotheses_embeddings,
                                                     targets=labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print("#batch:", len(data_loader))

        output_predictions = []
        golden_labels, predicted_labels = [], []
        for step, batch in enumerate(data_loader):
            b_premises = batch['premise']
            b_hypotheses = batch['hypothesis']

            # move tensors to GPU if CUDA is available
            b_premises = b_premises.to(device)
            b_hypotheses = b_hypotheses.to(device)

            # Compute the pair-wise cosine similarities
            start = time.monotonic()
            cos_similarity_scores, b_predictions = [], []
            for i in range(len(b_premises)):
                cosine_score = util.pytorch_cos_sim(b_premises[i], b_hypotheses[i]).cpu().detach().numpy()[0][0]
                cos_similarity_scores.append(cosine_score)
                predicted_label = ''
                for exp in label_map:
                    if eval(exp.replace('score', str(cosine_score))):
                        predicted_label = label_map[exp]
                        break
                b_predictions.append(predicted_label)

            end = time.monotonic()
            total_time += end - start
            print(f'time for calculating cosine similarity in step {step}: {end - start}')

            golden_labels.extend(batch['targets'])
            predicted_labels.extend(b_predictions)

            for i, item in enumerate(batch['item']):
                output_predictions.append((
                    premises[item],
                    hypotheses[item],
                    cos_similarity_scores[i],
                    batch['targets'][i],
                    b_predictions[i]
                ))

        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(premises))

        # evaluate
        print("Test Accuracy: {}".format(accuracy_score(golden_labels, predicted_labels)))
        print("Test Precision: {}".format(precision_score(golden_labels, predicted_labels, average="weighted")))
        print("Test Recall: {}".format(recall_score(golden_labels, predicted_labels, average="weighted")))
        print("Test F1-Score(weighted average): {}".format(
            f1_score(golden_labels, predicted_labels, average="weighted")))
        print("Test classification Report:\n{}".format(classification_report(
            golden_labels, predicted_labels, digits=10)))
        return output_predictions
