import os
import gc
import ast
import time
import hazm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForTokenClassification

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class NER:
    def __init__(self, model_name):
        self.normalizer = hazm.Normalizer()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        # self.labels = list(self.config.label2id.keys())
        self.id2label = self.config.id2label

    @staticmethod
    def load_ner_data(file_path, word_index, tag_index, delimiter, join=False):
        dataset, labels = [], []
        with open(file_path, encoding="utf8") as infile:
            sample_text, sample_label = [], []
            for line in infile:
                parts = line.strip().split(delimiter)
                if len(parts) > 1:
                    word, tag = parts[word_index], parts[tag_index]
                    if not word:
                        continue
                    sample_text.append(word)
                    sample_label.append(tag)
                else:
                    # end of sample
                    if sample_text and sample_label:
                        if join:
                            dataset.append(' '.join(sample_text))
                            labels.append(' '.join(sample_label))
                        else:
                            dataset.append(sample_text)
                            labels.append(sample_label)
                    sample_text, sample_label = [], []
        if sample_text and sample_label:
            if join:
                dataset.append(' '.join(sample_text))
                labels.append(' '.join(sample_label))
            else:
                dataset.append(sample_text)
                labels.append(sample_label)
        return dataset, labels

    def load_test_datasets(self, dataset_name, dataset_dir, **kwargs):
        if dataset_name.lower() == "peyma":
            ner_file_path = dataset_dir + 'test.txt'
            if not os.path.exists(ner_file_path):
                print(ner_file_path)
                exit(1)
            return self.load_ner_data(ner_file_path, word_index=0, tag_index=1, delimiter='|',
                                      join=kwargs.get('join', False))
        elif dataset_name.lower() == "arman":
            dataset, labels = [], []
            for i in range(1, 4):
                ner_file_path = dataset_dir + f'test_fold{i}.txt'
                if not os.path.exists(ner_file_path):
                    print(ner_file_path)
                dataset_part, labels_part = self.load_ner_data(ner_file_path, word_index=0, tag_index=1, delimiter=' ',
                                                               join=kwargs.get('join', False))
                dataset += dataset_part
                labels += labels_part
            return dataset, labels
        elif dataset_name.lower() == "hooshvare-peyman+arman+wikiann":
            ner_file_path = dataset_dir + 'test.csv'
            if not os.path.exists(ner_file_path):
                print(ner_file_path)
                exit(1)
            data = pd.read_csv(ner_file_path, delimiter="\t")
            sentences, sentences_tags = data['tokens'].values.tolist(), data['ner_tags'].values.tolist()
            sentences = [ast.literal_eval(ss) for ss in sentences]
            sentences_tags = [ast.literal_eval(ss) for ss in sentences_tags]
            print(f'test part:\n #sentences: {len(sentences)}, #sentences_tags: {len(sentences_tags)}')
            return sentences, sentences_tags

    def load_datasets(self, dataset_name, dataset_dir, **kwargs):
        if dataset_name.lower() == "farsiyar":
            dataset, labels = [], []
            for i in range(1, 6):
                ner_file_path = dataset_dir + 'Persian-NER-part{i}.txt'
                if not os.path.exists(ner_file_path):
                    print(ner_file_path)
                dataset_part, labels_part = self.load_ner_data(ner_file_path, word_index=0, tag_index=1, delimiter='\t',
                                                               join=kwargs.get('join', False))
                dataset += dataset_part
                labels += labels_part
            return dataset, labels
        elif dataset_name.lower() == "wikiann":
            ner_file_path = dataset_dir + 'wikiann-fa.bio'
            if not os.path.exists(ner_file_path):
                print(ner_file_path)
                exit(1)
            dataset_all, labels_all = self.load_ner_data(ner_file_path, word_index=0, tag_index=-1, delimiter=' ',
                                                         join=kwargs.get('join', False))
            print(f'all data: #data: {len(dataset_all)}, #labels: {len(labels_all)}')

            try:
                _, data_test, _, label_test = train_test_split(dataset_all, labels_all, test_size=0.1, random_state=1,
                                                               stratify=labels_all)
                print("with stratify")
            except:
                _, data_test, _, label_test = train_test_split(dataset_all, labels_all, test_size=0.1, random_state=1)
                print("without stratify")
            print(f'test part:\n #data: {len(data_test)}, #labels: {len(label_test)}')
            return dataset_all, labels_all, data_test, label_test

    def ner_inference(self, input_text, device, max_length):
        if not self.model or not self.tokenizer or not self.id2label:
            print('Something wrong has been happened!')
            return

        pt_batch = self.tokenizer(
            [self.normalizer.normalize(sequence) for sequence in input_text],
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

        pt_batch = pt_batch.to(device)
        pt_outputs = self.model(**pt_batch)
        pt_predictions = torch.argmax(pt_outputs.logits, dim=-1)
        pt_predictions = pt_predictions.cpu().detach().numpy().tolist()

        output_predictions = []
        for i, sequence in enumerate(input_text):
            tokens = self.tokenizer.tokenize(self.tokenizer.decode(self.tokenizer.encode(sequence)))
            predictions = [(token, self.id2label[prediction]) for token, prediction in
                           zip(tokens, pt_predictions[i])]
            output_predictions.append(predictions)
        return output_predictions

    def ner_evaluation(self, input_text, input_labels, device, batch_size=4):
        if not self.model or not self.tokenizer or not self.id2label:
            print('Something wrong has been happened!')
            return

        max_len = 0
        tokenized_texts, new_labels = [], []
        for sentence, sentence_label in zip(input_text, input_labels):
            if type(sentence) == str:
                sentence = sentence.strip().split()
            if len(sentence) != len(sentence_label):
                print('Something wrong has been happened! Length of a sentence and its label is not equal!')
                return
            tokenized_sentence, new_sentence_label = [], []
            for word, label in zip(sentence, sentence_label):
                # Tokenize the word and count # of subwords the word is broken into
                tokenized_word = self.tokenizer.tokenize(word)
                n_subwords = len(tokenized_word)

                # Add the tokenized word to the final tokenized word list
                tokenized_sentence.extend(tokenized_word)
                # Add the same label to the new list of labels `n_subwords` times
                new_sentence_label.extend([label] * n_subwords)

            max_len = max(max_len, len(tokenized_sentence))
            tokenized_texts.append(tokenized_sentence)
            new_labels.append(new_sentence_label)

        max_len = min(max_len, self.config.max_position_embeddings)
        print("max_len:", max_len)
        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=max_len, dtype="long", value=self.config.pad_token_id,
                                  truncating="post", padding="post")
        del tokenized_texts
        input_labels = pad_sequences([[self.config.label2id.get(l) for l in lab] for lab in new_labels],
                                     maxlen=max_len, value=self.config.label2id.get('O'), padding="post",
                                     dtype="long", truncating="post")
        del new_labels

        train_data = TensorDataset(torch.tensor(input_ids), torch.tensor(input_labels))
        data_loader = DataLoader(train_data, batch_size=batch_size)
        # data_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
        print("#samples:", len(input_ids))
        print("#batch:", len(data_loader))

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        total_loss, total_time = 0, 0
        output_predictions = []
        print("Start to evaluate test data ...")
        for step, batch in enumerate(data_loader):
            b_input_ids, b_labels = batch

            # move tensors to GPU if CUDA is available
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)

            # This will return the loss (rather than the model output) because we have provided the `labels`.
            with torch.no_grad():
                start = time.monotonic()
                outputs = self.model(b_input_ids, labels=b_labels)
                end = time.monotonic()
                total_time += end - start
                print(f'inference time for step {step}: {end - start}')
            # get the loss
            total_loss += outputs.loss.item()

            b_predictions = torch.argmax(outputs.logits, dim=2)
            b_predictions = b_predictions.cpu().detach().numpy().tolist()
            b_labels = b_labels.cpu().detach().numpy().tolist()

            for i, sample in enumerate(b_input_ids):
                sample_input = list(sample)
                # remove pad tokens
                while sample_input[-1] == self.config.pad_token_id:
                    sample_input.pop()
                # tokens = self.tokenizer.tokenize(self.tokenizer.decode(sample_input))
                tokens = [self.tokenizer.decode([t]) for t in sample_input]
                sample_true_labels = [self.id2label[e] for e in b_labels[i][:len(sample_input)]]
                sample_predictions = [self.id2label[e] for e in b_predictions[i][:len(sample_input)]]
                output_predictions.append(
                    [(t, sample_true_labels[j], sample_predictions[j]) for j, t in enumerate(tokens)])

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(data_loader)
        print("average loss:", avg_train_loss)
        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(input_ids))

        return output_predictions

    def ner_evaluation_2(self, input_text, input_labels, device, batch_size=4):
        if not self.model or not self.tokenizer or not self.id2label:
            print('Something wrong has been happened!')
            return

        print("len(input_text):", len(input_text))
        print("len(input_labels):", len(input_labels))
        c = 0
        max_len = 0
        tokenized_texts, new_labels = [], []
        for sentence, sentence_label in zip(input_text, input_labels):
            if type(sentence) == str:
                sentence = sentence.strip().split()
            if len(sentence) != len(sentence_label):
                print('Something wrong has been happened! Length of a sentence and its label is not equal!')
                return
            tokenized_words = self.tokenizer(sentence, padding=False, add_special_tokens=False).input_ids
            tokenized_sentence_ids, new_sentence_label = [], []
            for i, tokenized_word in enumerate(tokenized_words):
                # Add the tokenized word to the final tokenized word list
                tokenized_sentence_ids += tokenized_word
                # Add the same label to the new list of labels `number of subwords` times
                new_sentence_label.extend([self.config.label2id.get(sentence_label[i])] * len(tokenized_word))

            max_len = max(max_len, len(tokenized_sentence_ids))
            tokenized_texts.append(tokenized_sentence_ids)
            new_labels.append(new_sentence_label)
            c += 1
            if c % 10000 == 0:
                print("c:", c)
        max_len = min(max_len, self.config.max_position_embeddings)
        print("max_len:", max_len)
        input_ids = pad_sequences(tokenized_texts, maxlen=max_len, dtype="long", value=self.config.pad_token_id,
                                  truncating="post", padding="post")
        del tokenized_texts
        input_labels = pad_sequences(new_labels, maxlen=max_len, value=self.config.label2id.get('O'), padding="post",
                                     dtype="long", truncating="post")
        del new_labels

        train_data = TensorDataset(torch.tensor(input_ids), torch.tensor(input_labels))
        data_loader = DataLoader(train_data, batch_size=batch_size)
        # data_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
        print("#samples:", len(input_ids))
        print("#batch:", len(data_loader))

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        total_time = 0
        output_predictions = []
        print("Start to evaluate test data ...")
        for step, batch in enumerate(data_loader):
            b_input_ids, b_labels = batch

            # move tensors to GPU if CUDA is available
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)

            # This will return the loss (rather than the model output) because we have provided the `labels`.
            with torch.no_grad():
                start = time.monotonic()
                outputs = self.model(b_input_ids, labels=b_labels)
                end = time.monotonic()
                total_time += end - start
                print(f'inference time for step {step}: {end - start}')

            b_predictions = torch.argmax(outputs.logits, dim=2)
            b_predictions = b_predictions.cpu().detach().numpy().tolist()
            b_labels = b_labels.cpu().detach().numpy().tolist()

            for i, sample in enumerate(b_input_ids):
                sample_input = list(sample)
                # remove pad tokens
                while sample_input[-1] == self.config.pad_token_id:
                    sample_input.pop()
                # tokens = self.tokenizer.tokenize(self.tokenizer.decode(sample_input))
                tokens = [self.tokenizer.decode([t]) for t in sample_input]
                sample_true_labels = [self.id2label[e] for e in b_labels[i][:len(sample_input)]]
                sample_predictions = [self.id2label[e] for e in b_predictions[i][:len(sample_input)]]
                output_predictions.append(
                    [(t, sample_true_labels[j], sample_predictions[j]) for j, t in enumerate(tokens)])

        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(input_ids))

        return output_predictions

    def check_input_label_consistency(self, labels):
        model_labels = self.config.label2id.keys()
        dataset_labels = set()
        for l in labels:
            dataset_labels.update(set(l))
        print("model labels:", model_labels)
        print("dataset labels:", dataset_labels)
        print("intersection:", set(model_labels).intersection(dataset_labels))
        print("model_labels-dataset_labels:", list(set(model_labels) - set(dataset_labels)))
        print("dataset_labels-model_labels:", list(set(dataset_labels) - set(model_labels)))
        if list(set(dataset_labels) - set(model_labels)):
            return False
        return True

    @staticmethod
    def resolve_input_label_consistency(labels, label_translation_map):
        for i, sentence_labels in enumerate(labels):
            for j, label in enumerate(sentence_labels):
                labels[i][j] = label_translation_map.get(label)
        return labels

    @staticmethod
    def evaluate_prediction_results(labels, output_predictions):
        dataset_labels = set()
        for label in labels:
            dataset_labels.update(set(label))

        true_labels, predictions = [], []
        for sample_output in output_predictions:
            sample_true_labels = []
            sample_predicted_labels = []
            for token, true_label, predicted_label in sample_output:
                sample_true_labels.append(true_label)
                if predicted_label in dataset_labels:
                    sample_predicted_labels.append(predicted_label)
                else:
                    sample_predicted_labels.append('O')
            true_labels.append(sample_true_labels)
            predictions.append(sample_predicted_labels)

        print("Test Accuracy: {}".format(accuracy_score(true_labels, predictions)))
        print("Test Precision: {}".format(precision_score(true_labels, predictions)))
        print("Test Recall: {}".format(recall_score(true_labels, predictions)))
        print("Test F1-Score: {}".format(f1_score(true_labels, predictions)))
        print("Test classification Report:\n{}".format(classification_report(true_labels, predictions, digits=10)))
