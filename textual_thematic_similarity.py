import re
import gc
import os
import hazm
import time
import json
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification

from sentence_transformers import models, SentenceTransformer, util, evaluation

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

logging.basicConfig(level=logging.DEBUG)


class TextualThematicSimilarityDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Textual Thematic Similarity. """

    def __init__(self, sentences_1, sentences_2, targets, tokenizer, model_architecture, max_length):
        self.sentences_1 = sentences_1
        self.sentences_2 = sentences_2
        self.targets = targets
        self.tokenizer = tokenizer
        self.model_architecture = model_architecture
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences_1)

    def __getitem__(self, item):
        if self.model_architecture == "BertForSequenceClassification":
            encoding = self.tokenizer(
                [(self.sentences_1[item], self.sentences_2[item])],
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors="pt"
            )
            inputs = {
                'sentence_1': self.sentences_1[item],
                'sentence_2': self.sentences_2[item],
                'targets': self.targets[item],
                'input_ids': encoding.input_ids.flatten(),
                'attention_mask': encoding.attention_mask.flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten()
            }
            return inputs
        elif self.model_architecture == "sentence-transformer":
            inputs = {
                'item': item,
                'sentence_1': self.sentences_1[item],
                'sentence_2': self.sentences_2[item],
                'targets': self.targets[item]
            }
            return inputs
        return {}


class TextualThematicSimilarity:
    def __init__(self, model_name, model_architecture, label2id=None):
        self.normalizer = hazm.Normalizer()
        self.model_name = model_name
        self.model_architecture = model_architecture
        if self.model_architecture == "BertForSequenceClassification":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.label2id = label2id
            self.id2label = {i: l for l, i in label2id.items()}
        elif self.model_architecture == "sentence-transformer":
            word_embedding_model = models.Transformer(self.model_name)
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False)
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            self.config = AutoConfig.from_pretrained(self.model_name)

    def load_dataset_test_file(self, dataset_name, dataset_file, **kwargs):
        if dataset_name.lower() == "wiki-d-similar":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            data = pd.read_csv(dataset_file, delimiter="\t")

            # cleaning labels
            valid_labels = ['dissimilar', 'similar']
            data['Label'] = data['Label'].apply(lambda r: r if r in valid_labels else None)
            data = data.dropna(subset=['Label'])
            data = data.reset_index(drop=True)

            sentence1_list, sentence2_list = data['Sentence1'].values.tolist(), data['Sentence2'].values.tolist()
            labels = data['Label'].values.tolist()
            print(f'test part:\n #sentence1: {len(sentence1_list)}, #sentence2: {len(sentence2_list)}, '
                  f'#labels: {len(labels)}')
            return sentence1_list, sentence2_list, labels
        if dataset_name.lower() == "wiki-triplet":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            data = pd.read_csv(dataset_file, delimiter="\t")
            sentence1_list = data['Sentence1'].values.tolist()
            sentence2_list = data['Sentence2'].values.tolist()
            sentence3_list = data['Sentence3'].values.tolist()
            print(f'test part:\n #sentence1: {len(sentence1_list)}, #sentence2: {len(sentence2_list)}, '
                  f'#sentence3: {len(sentence3_list)}')
            return sentence1_list, sentence2_list, sentence3_list

    def thematic_similarity_inference_seq_classification(self, sentences_1, sentences_2, device, max_length):
        if not self.model or not self.tokenizer or not self.id2label:
            print('Something wrong has been happened!')
            return

        new_input = []
        for s1, s2 in zip(sentences_1, sentences_2):
            new_input.append((s1, s2))

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

        tokenized_batch = tokenized_batch.to(device)
        outputs = self.model(**tokenized_batch)
        pt_predictions = torch.argmax(F.softmax(outputs.logits, dim=1), dim=1)
        pt_predictions = pt_predictions.cpu().detach().numpy().tolist()

        output_predictions = []
        for i, sent1 in enumerate(sentences_1):
            output_predictions.append(
                (sent1, sentences_2[i], pt_predictions[i], self.id2label[pt_predictions[i]])
            )
        return output_predictions

    def thematic_similarity_inference_pair_similarity(self, sentences_1, sentences_2, device, label_list,
                                                      similarity_threshold=0.5):
        if not self.model:
            print('Something wrong has been happened!')
            return

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        # Compute the sentence embeddings
        sent1_embeddings = self.model.encode(sentences_1, convert_to_tensor=True, show_progress_bar=True)
        sent2_embeddings = self.model.encode(sentences_2, convert_to_tensor=True, show_progress_bar=True)

        # Compute the pair-wise cosine similarities
        similarity_scores, predicted_labels = [], []
        for i in range(len(sentences_1)):
            cos_scores = util.pytorch_cos_sim(sent1_embeddings[i], sent2_embeddings[i]).cpu().detach().numpy()
            similarity_scores.append(cos_scores[0][0])
            if cos_scores[0][0] >= similarity_threshold:
                predicted_labels.append(label_list[1])
            else:
                predicted_labels.append(label_list[0])

        output_predictions = []
        for i, sent1 in enumerate(sentences_1):
            output_predictions.append(
                (sent1, sentences_2[i], similarity_scores[i], predicted_labels[i])
            )
        return output_predictions

    def evaluation_seq_classification(self, sentence1_list, sentence2_list, labels, device, max_length, batch_size=4):
        if not self.model or not self.tokenizer or not self.id2label:
            print('Something wrong has been happened!')
            return
        label_count = {label: labels.count(label) for label in labels}
        print("label_count:", label_count)

        # convert labels
        new_labels = [self.label2id[_] for _ in labels]
        dataset = TextualThematicSimilarityDataset(sentences_1=sentence1_list, sentences_2=sentence2_list,
                                                   targets=new_labels, tokenizer=self.tokenizer,
                                                   model_architecture=self.model_architecture, max_length=max_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print(f'#sentence1:{len(sentence1_list)}, #sentence2:{len(sentence2_list)}, #labels:{len(labels)}')
        print("#batch:", len(data_loader))

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        total_loss, total_time = 0, 0
        output_predictions = []
        golden_labels, predicted_labels = [], []
        print("Start to evaluate test data ...")
        for step, batch in enumerate(data_loader):
            b_sentence_1 = batch['sentence_1']
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

            golden_labels.extend([self.id2label[_.item()] for _ in b_targets])

            b_predictions = torch.argmax(F.softmax(b_outputs.logits, dim=1), dim=1)
            b_predictions = b_predictions.cpu().detach().numpy().tolist()
            b_predictions = [self.id2label[_] for _ in b_predictions]
            predicted_labels.extend(b_predictions)

            for i, sent1 in enumerate(b_sentence_1):
                output_predictions.append((
                    sent1,
                    batch['sentence_2'][i],
                    self.id2label[b_targets[i].item()],
                    b_predictions[i]
                ))

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(data_loader)
        print("average loss:", avg_train_loss)
        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(sentence1_list))

        # evaluate
        print("Test Accuracy: {}".format(accuracy_score(golden_labels, predicted_labels)))
        print("Test Precision: {}".format(precision_score(golden_labels, predicted_labels, average="weighted")))
        print("Test Recall: {}".format(recall_score(golden_labels, predicted_labels, average="weighted")))
        print("Test F1-Score(weighted average): {}".format(
            f1_score(golden_labels, predicted_labels, average="weighted")))
        print("Test classification Report:\n{}".format(classification_report(
            golden_labels, predicted_labels, digits=10)))
        return output_predictions

    def evaluation_pair_similarity(self, sentence1_list, sentence2_list, labels, device, max_length, label_list,
                                   batch_size=4, similarity_threshold=0.5):
        if not self.model:
            print('Something wrong has been happened!')
            return
        label_count = {label: labels.count(label) for label in labels}
        print("label_count:", label_count)

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        total_time = 0
        print("Start to evaluate test data ...")

        # Compute the sentence embeddings
        start = time.monotonic()
        sent1_embeddings = self.model.encode(sentence1_list, convert_to_tensor=True, show_progress_bar=True,
                                             batch_size=batch_size)
        sent2_embeddings = self.model.encode(sentence2_list, convert_to_tensor=True, show_progress_bar=True,
                                             batch_size=batch_size)

        end = time.monotonic()
        total_time += end - start
        print(f'time for computing sentence embeddings: {end - start}')

        # # convert labels
        # new_labels = [self.label2id[_] for _ in labels]
        dataset = TextualThematicSimilarityDataset(sentences_1=sent1_embeddings, sentences_2=sent2_embeddings,
                                                   targets=labels, tokenizer=None,
                                                   model_architecture=self.model_architecture, max_length=max_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print(f'#sentence1:{len(sentence1_list)}, #sentence2:{len(sentence2_list)}, #labels:{len(labels)}')
        print("#batch:", len(data_loader))

        output_predictions = []
        golden_labels, predicted_labels = [], []
        for step, batch in enumerate(data_loader):
            b_sentence_1 = batch['sentence_1']
            b_sentence_2 = batch['sentence_2']

            # move tensors to GPU if CUDA is available
            b_sentence_1 = b_sentence_1.to(device)
            b_sentence_2 = b_sentence_2.to(device)

            # Compute the pair-wise cosine similarities
            start = time.monotonic()
            cos_similarity_scores, b_predictions = [], []
            for i in range(len(b_sentence_1)):
                cos_scores = util.pytorch_cos_sim(b_sentence_1[i], b_sentence_2[i]).cpu().detach().numpy()
                cos_similarity_scores.append(cos_scores[0][0])
                if cos_scores[0][0] >= similarity_threshold:
                    b_predictions.append(label_list[1])
                else:
                    b_predictions.append(label_list[0])
            end = time.monotonic()
            total_time += end - start
            print(f'time for calculating cosine similarity in step {step}: {end - start}')

            golden_labels.extend(batch['targets'])
            predicted_labels.extend(b_predictions)

            for i, item in enumerate(batch['item']):
                output_predictions.append((
                    sentence1_list[item],
                    sentence2_list[item],
                    cos_similarity_scores[i],
                    batch['targets'][i],
                    b_predictions[i]
                ))

        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(sentence1_list))

        # evaluate
        print("Test Accuracy: {}".format(accuracy_score(golden_labels, predicted_labels)))
        print("Test Precision: {}".format(precision_score(golden_labels, predicted_labels, average="weighted")))
        print("Test Recall: {}".format(recall_score(golden_labels, predicted_labels, average="weighted")))
        print("Test F1-Score(weighted average): {}".format(
            f1_score(golden_labels, predicted_labels, average="weighted")))
        print("Test classification Report:\n{}".format(classification_report(
            golden_labels, predicted_labels, digits=10)))
        return output_predictions

    def evaluation_pair_similarity_2(self, sentence1_list, sentence2_list, labels, device, label_list, batch_size=4):
        if not self.model:
            print('Something wrong has been happened!')
            return

        label_count = {label: labels.count(label) for label in labels}
        print("label_count:", label_count)
        new_labels = [label_list.index(l) for l in labels]

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        print("Start to evaluate test data ...")
        start = time.monotonic()
        evaluator = evaluation.BinaryClassificationEvaluator(
            sentences1=sentence1_list, sentences2=sentence2_list, labels=new_labels, name="Wiki d/similar",
            batch_size=batch_size, show_progress_bar=True, write_csv=True
        )
        output_scores = evaluator.compute_metrices(self.model)
        end = time.monotonic()
        print(f'total time: {end - start}')

        return output_scores

    def evaluation_triplet_similarity(self, sentence1_list, sentence2_list, sentence3_list, device, batch_size=4):
        """
        Given (sentence, positive_example, negative_example), checks if
        distance(sentence,positive_example) < distance(sentence, negative_example).
        """
        logging.basicConfig(level=logging.DEBUG)
        if not self.model:
            print('Something wrong has been happened!')
            return

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        print("Start to evaluate test data ...")
        start = time.monotonic()
        evaluator = evaluation.TripletEvaluator(
            anchors=sentence1_list, positives=sentence2_list, negatives=sentence3_list, name="wiki triplet",
            batch_size=batch_size, show_progress_bar=True, write_csv=True
        )
        output_scores = evaluator(self.model, output_path='.')
        end = time.monotonic()
        print(f'total time: {end - start}')

        return output_scores
