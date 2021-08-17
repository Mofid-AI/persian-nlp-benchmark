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


class SentimentAnalysisDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Sentiment Analysis. """

    def __init__(self, tokenizer, comments, targets, label_list=None, max_len=128):
        self.comments = comments
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2index = {label: i for i, label in enumerate(label_list)} if isinstance(label_list, list) else {}
        self.index2label = {i: label for label, i in self.label2index.items()}

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = self.comments[item]
        target = self.label2index[self.targets[item]]
        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt')

        inputs = {
            'comment': comment,
            'targets': torch.tensor(target, dtype=torch.long),
            'original_targets': self.targets[item],
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }

        return inputs


class MT5SentimentAnalysisDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Sentiment Analysis. """

    def __init__(self, reviews, aspects, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.aspects = aspects
        self.targets = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        if self.aspects is not None:
            encoding = self.tokenizer(
                self.reviews[item] + " <sep> " + self.aspects[item],
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors="pt"
            )
            inputs = {
                'review': self.reviews[item],
                'aspects': self.aspects[item],
                'targets': self.targets[item],
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
        else:
            encoding = self.tokenizer(
                self.reviews[item],
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors="pt"
            )
            inputs = {
                'review': self.reviews[item],
                'targets': self.targets[item],
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }

        return inputs


class SentimentAnalysis:
    def __init__(self, model_name, model_type=None):
        self.normalizer = hazm.Normalizer()
        self.model_name = model_name
        if model_type == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
            self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
            self.config = MT5Config.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.id2label = self.config.id2label
            self.label2id = self.config.label2id

    def cleaning(self, text):
        def cleanhtml(raw_html):
            clean_pattern = re.compile('<.*?>')
            clean_text = re.sub(clean_pattern, '', raw_html)
            return clean_text

        if type(text) is not str:
            return None

        text = text.strip()

        # regular cleaning
        text = clean(
            text,
            fix_unicode=True,
            to_ascii=False,
            lower=True,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=True,
            no_punct=False,
            replace_with_url="",
            replace_with_email="",
            replace_with_phone_number="",
            replace_with_number="",
            replace_with_digit="0",
            replace_with_currency_symbol=""
        )

        # cleaning htmls
        text = cleanhtml(text)

        # normalizing
        text = self.normalizer.normalize(text)

        # removing wierd patterns
        wierd_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u'\U00010000-\U0010ffff'
                                   u"\u200d"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\u3030"
                                   u"\ufe0f"
                                   u"\u2069"
                                   u"\u2066"
                                   # u"\u200c"
                                   u"\u2068"
                                   u"\u2067"
                                   "]+", flags=re.UNICODE)

        text = wierd_pattern.sub(r'', text)

        # removing extra spaces, hashtags
        text = re.sub("#", "", text)
        text = re.sub("\s+", " ", text)
        if text in ['', " "]:
            return None
        return text

    def load_dataset_test_file(self, dataset_name, dataset_file, **kwargs):
        if dataset_name.lower() == "snappfood":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            data = pd.read_csv(dataset_file, delimiter="\t")
            # drop label_id because its not consistent with albert model labels!
            data = data[['comment', 'label']]

            # cleaning comments
            data = data.dropna(subset=['comment'])
            data['comment'] = data['comment'].apply(self.cleaning)
            data = data.dropna(subset=['comment'])

            if 'label_map' in kwargs:
                data['label'] = data['label'].apply(lambda l: kwargs['label_map'][l])
                data = data.dropna(subset=['label'])
                data = data.reset_index(drop=True)

            data['label_id'] = data['label'].apply(lambda t: self.label2id[t])
            x_test, y_test = data['comment'].values.tolist(), data['label_id'].values.tolist()
            print(f'test part:\n #comment: {len(x_test)}, #labels: {len(y_test)}')
            return x_test, y_test
        if dataset_name.lower() == "deepsentipers":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            if 'label_map' not in kwargs:
                print("label_map is missing!")
                return
            data = pd.read_csv(dataset_file, delimiter=",", names=['comment', 'label'], header=None)

            # cleaning comments
            data = data.dropna(subset=['comment'])
            data['comment'] = data['comment'].apply(self.cleaning)
            data = data.dropna(subset=['comment'])

            # map labels
            label_map = kwargs['label_map']
            data['label'] = data['label'].apply(lambda l: label_map[l])
            data = data.dropna(subset=['label'])
            data = data.reset_index(drop=True)

            data['label_id'] = data['label'].apply(lambda t: self.label2id[t])
            x_test, y_test = data['comment'].values.tolist(), data['label_id'].values.tolist()
            print(f'test part:\n #comment: {len(x_test)}, #labels: {len(y_test)}')
            return x_test, y_test
        if dataset_name.lower() == "pasinlu-aspect-sentiment":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            if 'label_map' not in kwargs:
                print("label_map is missing!")
                return

            reviews, aspects, labels = [], [], []
            with open(dataset_file, encoding="utf8") as infile:
                for line in infile:
                    json_line = json.loads(line.strip())

                    review = json_line['review']
                    reviews.append(review)

                    question = json_line['question']
                    aspects.append(question)

                    label = kwargs['label_map'][json_line['label']]
                    labels.append(label)

            return reviews, aspects, labels

    def load_dataset_file(self, dataset_name, dataset_file, **kwargs):
        if dataset_name.lower() == "digikala":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            data = pd.read_excel(dataset_file)
            data = data[['comment', 'recommend']]

            # cleaning comments
            data = data.dropna(subset=['comment'])
            data['comment'] = data['comment'].apply(self.cleaning)
            data = data.dropna(subset=['comment'])

            # cleaning labels
            valid_labels = ['no_idea', 'not_recommended', 'recommended']
            data['recommend'] = data['recommend'].apply(lambda r: r if r in valid_labels else None)
            data = data.dropna(subset=['recommend'])
            if 'label_map' in kwargs:
                data['recommend'] = data['recommend'].apply(lambda l: kwargs['label_map'][l])
            data = data.dropna(subset=['recommend'])
            data = data.reset_index(drop=True)

            data['label_id'] = data['recommend'].apply(lambda t: self.label2id[t])

            x_all, y_all = data['comment'].values.tolist(), data['label_id'].values.tolist()
            print(f'all data: #comment: {len(x_all)}, #labels: {len(y_all)}')

            _, test = train_test_split(data, test_size=0.1, random_state=1, stratify=data['recommend'])
            test = test.reset_index(drop=True)
            x_test, y_test = test['comment'].values.tolist(), test['label_id'].values.tolist()
            print(f'test part:\n #comment: {len(x_test)}, #labels: {len(y_test)}')
            return x_all, y_all, x_test, y_test
        if dataset_name.lower() == "pasinlu-review-sentiment":
            if not os.path.exists(dataset_file):
                print(f'{dataset_file} not exists!')
                return
            if 'label_map' not in kwargs:
                print("label_map is missing!")
                return

            reviews, labels = [], []
            with open(dataset_file, encoding="utf8") as infile:
                for line in infile:
                    json_line = json.loads(line.strip())

                    review = json_line['review']
                    reviews.append(review)

                    label = kwargs['label_map'][json_line['sentiment']]
                    labels.append(label)
            return reviews, labels

    def load_dataset_composite_file(self, dataset_name, dataset_files, **kwargs):
        if dataset_name.lower() == "digikala+snappfood+deepsentipers":
            if sorted(list(dataset_files.keys())) != ["deepsentipers", "digikala", "snappfood"]:
                print("dataset_files must contains path of all three datasets")
                return
            if 'label_map' not in kwargs:
                print("label_map is missing!")
                return
            elif sorted(list(kwargs['label_map'].keys())) != ["deepsentipers", "digikala", "snappfood"]:
                print("label_map must contains label_map for all three datasets!")
                return
            print("digikala dataset - we only use test set:")
            _, _, x_test_digi, y_test_digi = self.load_dataset_file('digikala', dataset_files['digikala'],
                                                                    label_map=kwargs['label_map']['digikala'])
            print("snappfood dataset:")
            x_test_snapp, y_test_snapp = self.load_dataset_test_file('snappfood', dataset_files['snappfood'],
                                                                     label_map=kwargs['label_map']['snappfood'])
            print("deepsentipers dataset:")
            x_test_senti, y_test_senti = self.load_dataset_test_file('deepsentipers', dataset_files['deepsentipers'],
                                                                     label_map=kwargs['label_map']['deepsentipers'])
            return x_test_digi + x_test_snapp + x_test_senti, y_test_digi + y_test_snapp + y_test_senti

    def sentiment_analysis_inference(self, input_text, device):
        if not self.model or not self.tokenizer or not self.id2label:
            print('Something wrong has been happened!')
            return

        pt_batch = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=self.config.max_position_embeddings,
            return_tensors="pt"
        )

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()
        self.model.eval()
        pt_batch = pt_batch.to(device)

        pt_outputs = self.model(**pt_batch)
        pt_predictions = torch.argmax(F.softmax(pt_outputs.logits, dim=1), dim=1)

        output_predictions = []
        for i, sentence in enumerate(input_text):
            output_predictions.append((sentence, self.id2label.get(pt_predictions[i].item())))
        return output_predictions

    def mt5_sentiment_analysis_inference(self, reviews, device):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return

        tokenized_batch = self.tokenizer(
            reviews,
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

    def mt5_aspect_sentiment_analysis_inference(self, reviews, aspects, device):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return

        new_input = []
        for r, a in zip(reviews, aspects):
            new_input.append(r + " <sep> " + a)

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

    def evaluation(self, input_text, input_labels, device, batch_size=4):
        if not self.model or not self.tokenizer or not self.id2label:
            print('Something wrong has been happened!')
            return

        max_len = self.config.max_position_embeddings
        label_list = list(set(input_labels))
        label_count = {self.id2label[label]: input_labels.count(label) for label in label_list}
        print("label_count:", label_count)
        dataset = SentimentAnalysisDataset(comments=input_text, targets=input_labels, tokenizer=self.tokenizer,
                                           max_len=max_len, label_list=label_list)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        print("#samples:", len(input_text))
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
            b_comments = batch['comment']
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
            golden_labels.extend(b_original_targets.tolist())

            b_predictions = torch.argmax(F.softmax(b_outputs.logits, dim=1), dim=1)
            b_predictions = b_predictions.cpu().detach().numpy().tolist()
            b_predictions = [dataset.index2label[label] for label in b_predictions]
            predicted_labels.extend(b_predictions)

            for i, comment in enumerate(b_comments):
                output_predictions.append((
                    comment,
                    self.id2label[b_original_targets[i].item()],
                    self.id2label[b_predictions[i]]
                ))
                # print(f'output prediction: {i},{comment},{self.id2label[b_original_targets[i].item()]},'
                #       f'{self.id2label[b_predictions[i]]}')

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(data_loader)
        print("average loss:", avg_train_loss)
        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(input_text))

        # evaluate
        print("Test Accuracy: {}".format(accuracy_score(golden_labels, predicted_labels)))
        print("Test Precision: {}".format(precision_score(golden_labels, predicted_labels, average="weighted")))
        print("Test Recall: {}".format(recall_score(golden_labels, predicted_labels, average="weighted")))
        print("Test F1-Score(weighted average): {}".format(
            f1_score(golden_labels, predicted_labels, average="weighted")))
        print("Test classification Report:\n{}".format(classification_report(
            golden_labels, predicted_labels, digits=10, target_names=[self.id2label[_] for _ in sorted(label_list)])))
        return output_predictions

    def mt5_sentiment_analysis_evaluation(self, reviews, labels, device, max_length, batch_size=4):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return
        if len(reviews) != len(labels):
            print('length of inputs and labels is not equal!!')
            return

        dataset = MT5SentimentAnalysisDataset(reviews=reviews, aspects=None, labels=labels, tokenizer=self.tokenizer,
                                              max_length=max_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print(f'#reviews:{len(reviews)}, #labels:{len(labels)}')
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
            # move tensors to GPU if CUDA is available
            b_input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)

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

            for i, review in enumerate(batch['review']):
                output_predictions.append((
                    review,
                    b_targets[i],
                    b_predictions[i]
                ))

        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(reviews))

        # evaluate
        print("Test Accuracy: {}".format(accuracy_score(golden_labels, predicted_labels)))
        print("Test Precision: {}".format(precision_score(golden_labels, predicted_labels, average="weighted")))
        print("Test Recall: {}".format(recall_score(golden_labels, predicted_labels, average="weighted")))
        print("Test F1-Score(weighted average): {}".format(
            f1_score(golden_labels, predicted_labels, average="weighted")))
        print("Test classification Report:\n{}".format(
            classification_report(golden_labels, predicted_labels, digits=10)))
        return output_predictions

    def mt5_aspect_sentiment_analysis_evaluation(self, reviews, aspects, labels, device, max_length, batch_size=4):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return
        if len(reviews) != len(labels):
            print('length of inputs and labels is not equal!!')
            return

        dataset = MT5SentimentAnalysisDataset(reviews=reviews, aspects=aspects, labels=labels, tokenizer=self.tokenizer,
                                              max_length=max_length)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        print(f'#reviews:{len(reviews)}, #aspects:{len(aspects)}, #labels:{len(labels)}')
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
            # move tensors to GPU if CUDA is available
            b_input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)

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

            for i, review in enumerate(batch['review']):
                output_predictions.append((
                    review,
                    batch['aspects'][i],
                    b_targets[i],
                    b_predictions[i]
                ))

        print("total inference time:", total_time)
        print("total inference time / #samples:", total_time / len(reviews))

        # evaluate
        print("Test Accuracy: {}".format(accuracy_score(golden_labels, predicted_labels)))
        print("Test Precision: {}".format(precision_score(golden_labels, predicted_labels, average="weighted")))
        print("Test Recall: {}".format(recall_score(golden_labels, predicted_labels, average="weighted")))
        print("Test F1-Score(weighted average): {}".format(
            f1_score(golden_labels, predicted_labels, average="weighted")))
        print("Test classification Report:\n{}".format(
            classification_report(golden_labels, predicted_labels, digits=10)))
        return output_predictions
