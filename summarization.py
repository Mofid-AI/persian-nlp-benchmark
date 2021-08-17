import gc
import os
import time
import json
import torch
import datasets
import transformers

from rouge import RougeScorer
from transformers import BertTokenizerFast, EncoderDecoderConfig, EncoderDecoderModel


class Summarization:
    def __init__(self, model_name, model_type):
        self.model_name = model_name
        self.model_type = model_type.lower()
        if self.model_type == "bert2bert":
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
            self.config = EncoderDecoderConfig.from_pretrained(model_name)
            self.model = EncoderDecoderModel.from_pretrained(model_name, config=self.config)
        else:
            print(f'model_type not supported!')
            return

    @staticmethod
    def load_dataset_test_file(dataset_name, dataset_path, **kwargs):
        if dataset_name.lower() == "wiki-summary-v1.0.0":
            if not os.path.exists(dataset_path):
                print(f'{dataset_path} not exists!')
                return
            test_set = datasets.load_dataset(dataset_path, '1.0.0', split='test', cache_dir=None)
            return test_set
        if dataset_name.lower() == "wiki-summary-v2.0.0":
            if not os.path.exists(dataset_path):
                print(f'{dataset_path} not exists!')
                return
            test_set = datasets.load_dataset(dataset_path, '2.0.0', split='test', cache_dir=None)
            return test_set
        if dataset_name.lower() == "news-headline-v1.0.0":
            if not os.path.exists(dataset_path):
                print(f'{dataset_path} not exists!')
                return
            test_set = datasets.load_dataset(dataset_path, '1.0.0', split='test', cache_dir=None)
            return test_set

    def bert2bert_summarization_inference(self, sequence_list, device, max_length=512):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return

        inputs = self.tokenizer(
            sequence_list,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        outputs = self.model.generate(input_ids, attention_mask=attention_mask)
        generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated

    def bert2bert_evaluation(self, input_data, target_column, device, max_length=512, batch_size=4):
        if not self.model or not self.tokenizer:
            print('Something wrong has been happened!')
            return

        def generate_summary(batch):
            # Tokenizer will automatically set [BOS] <text> [EOS] cut off at BERT max length 512
            inputs = self.tokenizer(
                batch["article"],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            outputs = self.model.generate(input_ids, attention_mask=attention_mask)

            # all special tokens including will be removed
            output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            batch["predicted_summary"] = output_str
            return batch

        gc.collect()
        torch.cuda.empty_cache()
        # Tell pytorch to run this model on the GPU.
        if device.type != 'cpu':
            self.model.cuda()

        start = time.monotonic()
        results = input_data.map(generate_summary, batched=True, batch_size=batch_size)
        end = time.monotonic()
        print(f'evaluation time: {end - start}')
        print("total evaluation time / #samples:", (end - start) / len(input_data))

        scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
        rouge_output = scorer.compute(predictions=results["predicted_summary"], references=results[target_column])
        for rouge_metric in rouge_output:
            print(rouge_metric, rouge_output[rouge_metric])
        return results
