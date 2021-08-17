# Persian NLP Benchmark
The repository aims to track existing natural language processing models and evaluate their performance on well-known datasets.
Currently, we are benchmarking [HuggingFace persian models](https://huggingface.co/models?filter=fa), but we plan to expand our scope in the future.
We categorize the HuggingFace models based on their respective tasks.
These tasks include machine translation, multiple choice question answering, question paraphrasing, reading comprehension, 
sentiment analysis, summarization, text classification, textual entailment, and named entity recognition. 

We evaluate each model against well-known datasets.
The evaluation is performed on the test set.
In the absence of a test set, 10% of the original dataset is selected as the test set and the model is evaluated on it.

In this repository, we have prepared a script for each task that includes preparing models, loading datasets, and evaluating them. 
For each model, there is a notebook whose name is a combination of the task name and the corresponding model name.
These notebooks include all the steps involved in downloading the model, sample inference, loading the dataset, and evaluating the model.

Along the way, we faced many challenges.
Most of the available models presented the results of their evaluation on one or more datasets, but they didn't release evaluation codes.
Sometimes a sample code was provided to test the model on one or a limited number of samples. 
However, such code is insufficient to assess the performance of these model on real-world datasets.
Therefore, we had to develop such a code from scratch, which would probably make the details of our work different from theirs, and might lead to different results. 
Our implementation might be different in terms of data loading, preprocessing, preparing data for each type of model, capturing evaluation results and assessing its performance.
The lack of an explicit test set for some datasets was also a big challenge for our evaluation. 

We would be happy if you would like to participate in this path.
Please send us a pull request with evaluation script and notebook similar to ours.

# Benchmark Results

## Machine Translation Task
Machine translation is the task of automatically converting source text in one language to text in another language.
Here, we address the issue of translating English texts into Persian and vice versa. 

### Sample Translation
```python
import torch
from machine_translation import MachineTranslation

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name='persiannlp/mt5-small-parsinlu-translation_en_fa'

mt_model = MachineTranslation(model_name=model_name, model_type="mt5")
input_list = [
  "Praise be to Allah, the Cherisher and Sustainer of the worlds;",
  "shrouds herself in white and walks penitentially disguised as brotherly love through factories and parliaments; offers help, but desires power;",
  "He thanked all fellow bloggers and organizations that showed support.",
  "Races are held between April and December at the Veliefendi Hippodrome near Bakerky, 15 km (9 miles) west of Istanbul.",
  "I want to pursue PhD in Computer Science about social network,what is the open problem in social networks?"
]
mt_model.mt5_machine_translation_inference(input_list, device)
```

### English to Persian Translation
We evaluated the available models on the following datasets: 

- [**Mizan**](https://media.githubusercontent.com/media/persiannlp/parsinlu/master/data/translation/mizan/mizan_test_en_fa.tsv): 
         It is a parallel corpora constructed from human translations of literary masterpieces. 
         This dataset contains 10000 test sentence pairs. 
- [**Quran**](https://media.githubusercontent.com/media/persiannlp/parsinlu/master/data/translation/quran/quran_en_fa.tsv): 
         It is based on the existing translations of Quran. This dataset contains 6236 english sentence and multiple persian translate. 
         Since no test set is explicitly placed for this dataset, we evaluated the available models against the entire data as well as 10 percent of it as a test set.
         It should be noted that the evaluation of translations of Quran samples was done once by considering a single string of all Persian translations and once by considering a set containing Persian translations. 
- [**Bible**](https://media.githubusercontent.com/media/persiannlp/parsinlu/master/data/translation/bible/bible_en_fa.tsv): 
         It is based on the existing translations of Bible. This dataset contains 31020 sentence pairs. 
         Since no test set is explicitly placed for this dataset, we evaluated the available models against the entire data as well as 10 percent of it as a test set.
- [**Combined**](https://media.githubusercontent.com/media/persiannlp/parsinlu/master/data/translation/translation_combined_en_fa/test.tsv): 
            This dataset combines four datasets, including Mizan, Quranو Bible, and QQP.
            It contains 48123 records, of which 31020, 10000, 1104, and 5999 samples are taken from the Bible, Mizan, QQP, and Quran, respectively.
            We evaluate the whole data as well as each subset separately. 
            It should be noted that the evaluation of translations of Quran samples was done once by considering a single string of all Persian translations and once by considering a set containing Persian translations.

All evaluation steps can be found in the [notebooks](notebooks/machine-translation) associated with this task.
The evaluation metric for this task is the BLEU score. 
This metric is calculated for all experiments, and the results are aggregated in the corresponding [result file](evaluation-results/machine_translation.xlsx).
This file contains information such as the hardware, the time taken for the evaluation, and the final results.

In the following table, we will report BLEU score for each subset of combined dataset of ParsiNLU:

|  Notebook                                                                                                    |  Model Type   |        Model Name                               | Quran | Bible | Mizan |  QQP   |
|:------------------------------------------------------------------------------------------------------------:|:-------------:|:-----------------------------------------------:|:-----:|:-----:|:-----:|:------:|
|[Link](notebooks/machine-translation/MachineTranslation_persiannlp_mt5-small-parsinlu-translation_en_fa.ipynb)|  mT5 (small)  | persiannlp/mt5-small-parsinlu-translation_en_fa | 4.232 | 0.173 | 3.958 | 16.473 |
|[Link](notebooks/machine-translation/MachineTranslation_persiannlp_mt5-base-parsinlu-translation_en_fa.ipynb) |  mT5 (base)   | persiannlp/mt5-base-parsinlu-translation_en_fa  | 5.166 | 0.216 | 4.957 | 19.972 |
|[Link](notebooks/machine-translation/MachineTranslation_persiannlp_mt5-large-parsinlu-translation_en_fa.ipynb)|  mT5 (large)) | persiannlp/mt5-large-parsinlu-translation_en_fa | 5.735 | 0.220 | 5.545 | 21.645 |

### Persian to English
We evaluated the available models on the following datasets: 

- [**Mizan**](https://media.githubusercontent.com/media/persiannlp/parsinlu/master/data/translation/mizan/mizan_test_fa_en.tsv): 
         It is a parallel corpora constructed from human translations of literary masterpieces. 
         This dataset contains 10000 test sentence pairs. 
- [**Quran**](https://media.githubusercontent.com/media/persiannlp/parsinlu/master/data/translation/quran/quran_fa_en.tsv): 
         It is based on the existing translations of Quran. This dataset contains 6236 english sentence and multiple persian translate. 
         Since no test set is explicitly placed for this dataset, we evaluated the available models against the entire data as well as 10 percent of it as a test set.
         It should be noted that the evaluation of translations of Quran samples was done once by considering a single string of all Persian translations and once by considering a set containing Persian translations. 
- [**Bible**](https://media.githubusercontent.com/media/persiannlp/parsinlu/master/data/translation/bible/bible_fa_en.tsv): 
         It is based on the existing translations of Bible. This dataset contains 31020 sentence pairs. 
         Since no test set is explicitly placed for this dataset, we evaluated the available models against the entire data as well as 10 percent of it as a test set.
- [**Combined**](https://media.githubusercontent.com/media/persiannlp/parsinlu/master/data/translation/translation_combined_fa_en/test.tsv): 
            This dataset combines four datasets, including Mizan, Quranو Bible, and QQP.
            It contains 47738 records, of which 31020, 10000, 489, and 6229 samples are taken from the Bible, Mizan, QQP, and Quran, respectively.
            We evaluate the whole data as well as each subset separately. 
            It should be noted that the evaluation of translations of Quran samples was done once by considering a single string of all Persian translations and once by considering a set containing Persian translations.

All evaluation steps can be found in the [notebooks](notebooks/machine-translation) associated with this task.
The evaluation metric for this task is the BLEU score. 
This metric is calculated for all experiments, and the results are aggregated in the corresponding [result file](evaluation-results/machine_translation.xlsx).
This file contains information such as the hardware, the evaluation time, and the final results.

In the following table, we will report BLEU score for each subset of combined dataset of ParsiNLU:  

|  Notebook                                                                                                         |  Model Type   |             Model Name                               | Quran  | Bible | Mizan  |  QQP   |
|:-----------------------------------------------------------------------------------------------------------------:|:-------------:|:----------------------------------------------------:|:------:|:-----:|:------:|:------:|
|[Link](notebooks/machine-translation/MachineTranslation_persiannlp_mt5-small-parsinlu-opus-translation_fa_en.ipynb)|  mT5 (small)  | persiannlp/mt5-small-parsinlu-opus-translation_fa_en | 7.443  | 0.367 | 8.425  | 21.809 |
|[Link](notebooks/machine-translation/MachineTranslation_persiannlp_mt5-base-parsinlu-opus-translation_fa_en.ipynb) |  mT5 (base)   | persiannlp/mt5-base-parsinlu-opus-translation_fa_en  | 9.253  | 0.376 | 9.848  | 26.898 |
|[Link](notebooks/machine-translation/MachineTranslation_persiannlp_mt5_large_parsinlu_opus_translation_fa_en.ipynb)|  mT5 (large)) | persiannlp/mt5-large-parsinlu-opus-translation_fa_en | 11.650 | 0.458 | 12.332 | 30.414 |

## Multiple Choice Question Answering Task
Given a natural language question, this task aims to pick the correct answer among a list of multiple candidates. 
A key difference from reading comprehension is that the instances are open-domain (i.e., no context paragraph is provided). 
Hence, a model would either need to retrieve supporting documents from an external source, or have stored the necessary knowledge internally to be able to answer such QAs.

### Sample Inference
```python
import torch
from multiple_choice_qa import MultipleChoiceQA

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name='persiannlp/mt5-small-parsinlu-multiple-choice'
mcqa_model = MultipleChoiceQA(model_name=model_name, model_type="mt5")

question_list = [
    "وسیع ترین کشور جهان کدام است؟",
    "طامع یعنی ؟",
    "زمینی به ۳۱ قطعه متساوی مفروض شده است و هر روز مساحت آماده شده برای احداث، دو برابر مساحت روز قبل است.اگر پس از (۵ روز) تمام زمین آماده شده باشد، در چه روزی یک قطعه زمین آماده شده"
]
candidate_list=[
    ["آمریکا", "کانادا", "روسیه", "چین"],
    ["آزمند", "خوش شانس", "محتاج", "مطمئن"],
    ["روز اول", "روز دوم", "روز سوم", "هیچکدام"]
]
mcqa_model.mt5_multiple_choice_qa_inference(question_list, candidate_list, device)
```

### Evaluation
We evaluated the available models on the following datasets: 

- [**ParsiNLU - literature**](https://raw.githubusercontent.com/persiannlp/parsinlu/master/data/multiple-choice/test_lit.jsonl): 
               This dataset contains 350 test questions.
               These questions are from the annual college entrance exams in Iran, for the past 15 years. 
               They questions often involve understanding poetry and their implied meaning, knowledge of Persian grammar, and the history of literature.
- [**ParsiNLU - math & logic**](https://raw.githubusercontent.com/persiannlp/parsinlu/master/data/multiple-choice/test_ml.jsonl): 
               This dataset contains 350 test questions.
               These questions are from employment exams that are expected to assess individual’s depth in various topics (accounting, teaching, mathematics, logic, etc).
- [**ParsiNLU - common knowledge**](https://raw.githubusercontent.com/persiannlp/parsinlu/master/data/multiple-choice/test_ck.jsonl): 
               This dataset contains 350 test questions.
               These questions are common knowledge questions, which involve questions about topics such as basic science, history, or geography.

All evaluation steps can be found in the [notebooks](notebooks/multiple-choice-qa) associated with this task.
All the experimental results are aggregated in the corresponding [result file](evaluation-results/multiple_choice_qa.xlsx).
This file contains information such as the hardware, the evaluation time, and the final results.

In the following table, we will report evaluation results for `literature` dataset from ParsiNLU:

|  Notebook                                                                                                              |  Model Type   |             Model Name                                      | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) | Exact String Match Score | F1 String Match Score | 
|:----------------------------------------------------------------------------------------------------------------------:|:-------------:|:-----------------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|:------------------------:|:---------------------:|
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-small-parsinlu-multiple-choice.ipynb)               |  mT5 (small)  | persiannlp/mt5-small-parsinlu-multiple-choice               |  38.286  |       38.288        |      37.954      |      38.286      |     37.672    |       38.067        |      37.608      |           33.429         |         45.452        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-base-parsinlu-multiple-choice.ipynb)                |  mT5 (base)   | persiannlp/mt5-base-parsinlu-multiple-choice                |  38.571  |       38.585        |      38.266      |      38.571      |     38.150    |       38.555        |      38.184      |           33.429         |         45.451        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-large-parsinlu-multiple-choice.ipynb)               |  mT5 (large)  | persiannlp/mt5-large-parsinlu-multiple-choice               |  37.429  |       37.565        |      37.042      |      37.429      |     37.069    |       37.482        |      37.040      |           32.286         |         44.165        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-small-parsinlu-arc-comqa-obqa-multiple-choice.ipynb)|  mT5 (small)  | persiannlp/mt5-small-parsinlu-arc-comqa-obqa-multiple-choice|  34.571  |       34.502        |      33.941      |      34.571      |     33.997    |       34.461        |      33.896      |           29.714         |         41.866        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-base-parsinlu-arc-comqa-obqa-multiple-choice.ipynb) |  mT5 (base)   | persiannlp/mt5-base-parsinlu-arc-comqa-obqa-multiple-choice |  37.143  |       36.928        |      36.641      |      37.143      |     36.776    |       36.976        |      36.653      |           32.000         |         44.052        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-large-parsinlu-arc-comqa-obqa-multiple-choice.ipynb)|  mT5 (large)  | persiannlp/mt5-large-parsinlu-arc-comqa-obqa-multiple-choice|  35.143  |       35.168        |      34.702      |      35.143      |     35.012    |       35.042        |      34.741      |           30.286         |         42.141        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mbert-base-parsinlu-multiple-choice.ipynb)              |  mBert        | persiannlp/mbert-base-parsinlu-multiple-choice              |  21.429  |        4.592        |       5.357      |      21.429      |     25.000    |        7.563        |       8.824      |             -            |            -          |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_wikibert-base-parsinlu-multiple-choice.ipynb)           |  WikiBert     | persiannlp/wikibert-base-parsinlu-multiple-choice           |  21.429  |        4.592        |       5.357      |      21.429      |     25.000    |        7.563        |       8.824      |             -            |            -          |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_parsbert-base-parsinlu-multiple-choice.ipynb)           |  ParsBert     | persiannlp/parsbert-base-parsinlu-multiple-choice           |  20.571  |       20.554        |      20.392      |      20.571      |     22.512    |       15.226        |      15.694      |             -            |            -          |

In the following table, we will report evaluation results for `math & logic` dataset from ParsiNLU:

|  Notebook                                                                                                              |  Model Type   |             Model Name                                      | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) | Exact String Match Score | F1 String Match Score | 
|:----------------------------------------------------------------------------------------------------------------------:|:-------------:|:-----------------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|:------------------------:|:---------------------:|
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-small-parsinlu-multiple-choice.ipynb)               |  mT5 (small)  | persiannlp/mt5-small-parsinlu-multiple-choice               |  44.286  |       45.291        |      43.861      |      44.286      |     43.268    |       44.447        |       43.226     |           40.857         |         46.083        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-base-parsinlu-multiple-choice.ipynb)                |  mT5 (base)   | persiannlp/mt5-base-parsinlu-multiple-choice                |  40.571  |       41.556        |      39.816      |      40.571      |     39.883    |       40.823        |       39.633     |           37.429         |         42.493        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-large-parsinlu-multiple-choice.ipynb)               |  mT5 (large)  | persiannlp/mt5-large-parsinlu-multiple-choice               |  42.000  |       42.513        |      40.507      |      42.000      |     40.698    |       42.177        |       40.533     |           40.857         |         45.611        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-small-parsinlu-arc-comqa-obqa-multiple-choice.ipynb)|  mT5 (small)  | persiannlp/mt5-small-parsinlu-arc-comqa-obqa-multiple-choice|  39.714  |       40.853        |      38.603      |      39.714      |     38.744    |       40.073        |       38.476     |           36.857         |         43.334        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-base-parsinlu-arc-comqa-obqa-multiple-choice.ipynb) |  mT5 (base)   | persiannlp/mt5-base-parsinlu-arc-comqa-obqa-multiple-choice |  41.143  |       42.395        |      40.717      |      41.143      |     41.094    |       41.360        |       40.542     |           38.857         |         44.555        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-large-parsinlu-arc-comqa-obqa-multiple-choice.ipynb)|  mT5 (large)  | persiannlp/mt5-large-parsinlu-arc-comqa-obqa-multiple-choice|  40.571  |       41.530        |      39.606      |      40.571      |     40.273    |       40.801        |       39.685     |           39.143         |         44.455        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mbert-base-parsinlu-multiple-choice.ipynb)              |  mBert        | persiannlp/mbert-base-parsinlu-multiple-choice              |  33.714  |       11.367        |       8.429      |      33.714      |     25.000    |       17.001        |       12.607     |             -            |            -          |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_wikibert-base-parsinlu-multiple-choice.ipynb)           |  WikiBert     | persiannlp/wikibert-base-parsinlu-multiple-choice           |  33.714  |       11.367        |       8.429      |      33.714      |     25.000    |       17.001        |       12.607     |             -            |            -          |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_parsbert-base-parsinlu-multiple-choice.ipynb)           |  ParsBert     | persiannlp/parsbert-base-parsinlu-multiple-choice           |  32.286  |       29.745        |      28.279      |      32.286      |     25.694    |       23.933        |       20.478     |             -            |            -          |

In the following table, we will report evaluation results for `common knowledge` dataset from ParsiNLU:

|  Notebook                                                                                                              |  Model Type   |             Model Name                                      | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) | Exact String Match Score | F1 String Match Score |
|:----------------------------------------------------------------------------------------------------------------------:|:-------------:|:-----------------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|:------------------------:|:---------------------:|
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-small-parsinlu-multiple-choice.ipynb)               |  mT5 (small)  | persiannlp/mt5-small-parsinlu-multiple-choice               |  26.286  |       26.542        |       26.331     |      26.286      |     26.383    |       26.332        |      26.266      |           24.000         |         34.471        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-base-parsinlu-multiple-choice.ipynb)                |  mT5 (base)   | persiannlp/mt5-base-parsinlu-multiple-choice                |  24.571  |       24.580        |       24.348	   |      24.571      |     24.499    |       24.552        |      24.397      |           23.429         |         34.014        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-large-parsinlu-multiple-choice.ipynb)               |  mT5 (large)  | persiannlp/mt5-large-parsinlu-multiple-choice               |  27.429  |       27.914        |       27.390	   |      27.429      |     27.180    |       27.613        |      27.218      |           27.143         |         36.625        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-small-parsinlu-arc-comqa-obqa-multiple-choice.ipynb)|  mT5 (small)  | persiannlp/mt5-small-parsinlu-arc-comqa-obqa-multiple-choice|  27.143  |       27.330        |       27.153	   |      27.143      |     27.104    |       27.199        |      27.091      |           23.429         |         34.945        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-base-parsinlu-arc-comqa-obqa-multiple-choice.ipynb) |  mT5 (base)   | persiannlp/mt5-base-parsinlu-arc-comqa-obqa-multiple-choice |  25.143  |       25.230        |       24.819	   |      25.143      |     24.700    |       25.155        |      24.730      |           23.143         |         34.040        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mt5-large-parsinlu-arc-comqa-obqa-multiple-choice.ipynb)|  mT5 (large)  | persiannlp/mt5-large-parsinlu-arc-comqa-obqa-multiple-choice|  30.000  |       29.970        |       29.677	   |      30.000      |     29.640    |       29.963	    |      29.636      |           29.143         |         39.102        |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_mbert-base-parsinlu-multiple-choice.ipynb)              |  mBert        | persiannlp/mbert-base-parsinlu-multiple-choice              |  28.000  |        7.862        |        7.020     |      28.000      |     25.000    |       12.277        |      10.962      |             -            |            -          |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_wikibert-base-parsinlu-multiple-choice.ipynb)           |  WikiBert     | persiannlp/wikibert-base-parsinlu-multiple-choice           |  28.000  |        7.840        |        7.000     |      28.000      |     25.000    |       12.250        |      10.937      |             -            |            -          |
|[Link](notebooks/multiple-choice-qa/MultipleChoiceQA_persiannlp_parsbert-base-parsinlu-multiple-choice.ipynb)           |  ParsBert     | persiannlp/parsbert-base-parsinlu-multiple-choice           |  26.857  |       25.297        |       24.319     |      26.857      |     24.614    |       20.749        |      19.419      |             -            |            -          |

## Question Paraphrasing Task
This task aims to detect whether two given questions are paraphrases of each other or not.
For a given pair of natural-language questions, one must determine whether they are paraphrases or not. 
Paraphrasing has a broad range of applications and, in particular, query-paraphrasing can be used to improve document retrieval. 

### Sample Inference
```python
import torch
from question_paraphrasing import QuestionParaphrasing

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name='persiannlp/mt5-small-parsinlu-qqp-query-paraphrasing'
qp_model = QuestionParaphrasing(model_name=model_name, model_type="mt5")

q1_list = [
  "چه چیزی باعث پوکی استخوان می شود؟", 
  "من دارم به این فکر میکنم چرا ساعت هفت نمیشه؟", 
  "دعای کمیل در چه روزهایی خوانده می شود؟", 
  "دعای کمیل در چه روزهایی خوانده می شود؟",
  "شناسنامه در چه سالی وارد ایران شد؟",
  "سیب زمینی چه زمانی وارد ایران شد؟"
]
q2_list = [
  "چه چیزی باعث مقاومت استخوان در برابر ضربه می شود؟", 
  "چرا من ساده فکر میکردم به عشقت پابندی؟", 
  "دعای جوشن کبیر در چه شبی خوانده می شود؟", 
  "دعای جوشن کبیر در چه شبی خوانده می شود؟",
  "سیب زمینی در چه سالی وارد ایران شد؟",
  "سیب زمینی در چه سالی وارد ایران شد؟"
]
qp_model.mt5_question_paraphrasing_inference(q1_list, q2_list, device)
```

### Evaluation
We evaluated the available models on the following datasets: 

- [**ParsiNLU - QQP**](https://raw.githubusercontent.com/persiannlp/parsinlu/master/data/qqp/test.jsonl): 
               This dataset contains 1916 test question pairs. There are two type of samples: 
               478 test question pairs with qqp category and 1438 test question pairs with natural category.
               QQP questions are from existing QQP English dataset. They translate it with Google Translate API. 
               Afterwards, expert annotators carefully re-annotate the result of the translations to fix any inaccuracies.
               Natural question are mined using Google auto-complete as well as an additional set of questions mined from Persian discussion forums. 
               They create pairs of questions with high token overlap. Each pair is annotated by native-speaker annotator as paraphrase or not-paraphrase. 
               They drop the pair if any of the questions is incomplete.            

All evaluation steps can be found in the [notebooks](notebooks/question-paraphrasing) associated with this task.
All the experimental results are aggregated in the corresponding [result file](evaluation-results/question_paraphrasing.xlsx).
This file contains information such as the hardware, the evaluation time, and the final results.

In the following table, we will report evaluation results for `qqp` subset of ParsiNLU - QQP dataset:

|  Notebook                                                                                                               |  Model Type   |             Model Name                                 | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:-----------------------------------------------------------------------------------------------------------------------:|:-------------:|:------------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/question-paraphrasing/QuestionParaphrasing_parsinlu-mt5_mt5-small-parsinlu-qqp-query-paraphrasing.ipynb)|  mT5 (small)  | parsinlu-mt5/mt5-small-parsinlu-qqp-query-paraphrasing |  71.967  |        72.100       |       70.064     |      71.967      |     70.242    |       72.028        |      70.147      |
|[Link](notebooks/question-paraphrasing/QuestionParaphrasing_parsinlu-mt5_mt5-base-parsinlu-qqp-query-paraphrasing.ipynb) |  mT5 (base)   | parsinlu-mt5/mt5-base-parsinlu-qqp-query-paraphrasing  |  74.268  |        74.131       |       72.456     |      74.268      |     72.189    |       74.191        |      72.313      |
|[Link](notebooks/question-paraphrasing/QuestionParaphrasing_parsinlu-mt5_mt5-large-parsinlu-qqp-query-paraphrasing.ipynb)|  mT5 (large)  | parsinlu-mt5/mt5-large-parsinlu-qqp-query-paraphrasing |  77.824  |        77.993       |       76.279     |      77.824      |     76.622    |       77.896        |      76.437      |

In the following table, we will report evaluation results for `natural` subset of ParsiNLU - QQP dataset:

|  Notebook                                                                                                               |  Model Type   |             Model Name                                 | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:-----------------------------------------------------------------------------------------------------------------------:|:-------------:|:------------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/question-paraphrasing/QuestionParaphrasing_parsinlu-mt5_mt5-small-parsinlu-qqp-query-paraphrasing.ipynb)|  mT5 (small)  | parsinlu-mt5/mt5-small-parsinlu-qqp-query-paraphrasing |  77.886  |       78.284        |       78.508     |      77.886      |     77.027    |       77.593        |      77.263      |
|[Link](notebooks/question-paraphrasing/QuestionParaphrasing_parsinlu-mt5_mt5-base-parsinlu-qqp-query-paraphrasing.ipynb) |  mT5 (base)   | parsinlu-mt5/mt5-base-parsinlu-qqp-query-paraphrasing  |  79.416  |       79.447        |       79.484     |      79.416      |     78.925    |       79.318        |      79.088      |
|[Link](notebooks/question-paraphrasing/QuestionParaphrasing_parsinlu-mt5_mt5-large-parsinlu-qqp-query-paraphrasing.ipynb)|  mT5 (large)  | parsinlu-mt5/mt5-large-parsinlu-qqp-query-paraphrasing |  85.396  |       85.428        |       85.481     |      85.396      |     85.050    |       85.352        |      85.204      |

## Reading Comprehension Task
In this task, the goal is to generate a response to question and its accompanying context paragraph. 
We use the commonly used definition of reading comprehension task: generating an answer, given a question and a context paragraph.

### Sample Inference
```python
import torch
from reading_comprehension import ReadingComprehension

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name='persiannlp/mt5-small-parsinlu-squad-reading-comprehension'
rc_model = ReadingComprehension(model_name=model_name, model_type="mt5")

context_list = [
    "یک شی را دارای تقارن می‌نامیم زمانی که ان شی را بتوان به دو یا چند قسمت تقسیم کرد که آن‌ها قسمتی از یک طرح سازمان یافته باشند یعنی بر روی شکل تنها جابجایی و چرخش و بازتاب و تجانس انجام شود و در اصل شکل تغییری به وجود نیایید آنگاه ان را تقارن می‌نامیم مرکز تقارن:اگر در یک شکل نقطه‌ای مانندA وجود داشته باشد که هر نقطهٔ روی شکل (محیط) نسبت به نقطه یAمتقارن یک نقطهٔ دیگر شکل (محیط) باشد، نقطهٔ Aمرکز تقارن است. یعنی هر نقطه روی شکل باید متقارنی داشته باشد شکل‌های که منتظم هستند و زوج ضلع دارند دارای مرکز تقارند ولی شکل‌های فرد ضلعی منتظم مرکز تقارن ندارند. متوازی‌الأضلاع و دایره یک مرکز تقارن دارند ممکن است یک شکل خط تقارن نداشته باشد ولی مرکز تقارن داشته باشد. (منبع:س. گ)",
    "شُتُر یا اُشتر را که در زبان پهلوی (ushtar)[نیازمند منبع] می‌گفتند حیوانی است نیرومند و تنومند با توش و توان بالا از خانواده شتران؛ شبه نشخوارکننده و با دست و گردنی دراز. بر پشت خود یک یا دو کوهان دارد که ساختارش از پیه و چربی است. در دین اسلام گوشت او حلال است. اما ذبح آن با دیگر جانوران حلال گوشت متفاوت است و آن را نحر (بریدن گلو) می‌کنند و اگر سر آن را مانند گوسفند پیش از نحر ببرند گوشت آن حلال نیست. شیرش نیز نوشیده می‌شود ولی بیشتر کاربرد بارکشی دارد. پشم و پوستش نیز برای ریسندگی و پارچه‌بافی و کفش‌دوزی کاربرد دارد.  گونه‌های دیگری از شتران نیز در آمریکای جنوبی زندگی می‌کنند، به نام‌های لاما، آلپاکا، گواناکو که دارای کوهان نیستند.  شتر ویژگی‌های خاصّی دارد که مهم‌ترین آن‌ها تحمّل شرایط سخت صحرا و دماهای گوناگون و به‌ویژه گرمای شدید تابستان و کمبود آب و علوفه است. ترکیب جسمانی شتر با دیگر جانوران اختلاف زیادی دارد، و این اختلاف انگیزه شده که شتر در درازا روزهای سال در بیابان زندگی کند و از بوته‌ها و درختچه‌های گوناگون صحرایی و کویری و حتی از بوته‌های شور و خاردار تغذیه کند. عرب‌ها از زمان‌های بسیار دور از شتر استفاده کرده و می‌کنند. آن‌ها به این حیوان اهلی لقب کشتی صحرا (به عربی: سفینةالصحراء) داده‌اند.",
    """حسین میرزایی می‌گوید مرحله اول پرداخت وام حمایتی کرونا به همگی خانوارهای یارانه‌بگیر متقاضی تکمیل شده است و حال چهار میلیون خانوار که به عنوان "اقشار خاص" و "آسیب‌پذیر" شناسایی شدند، می‌توانند برای یک میلیون تومان وام دیگر درخواست بدهند. آقای میرزایی گفته خانوارهای "آسیب‌پذیر" که شرایط گرفتن وام یک میلیونی اضافی را دارند با پیامک از این امکان مطلع شده‌اند. بنا به گزارش‌های رسمی با شیوع کرونا در ایران یک میلیون نفر بیکار شده‌اند و درآمد کارکنان مشاغل غیررسمی نیز ضربه قابل توجهی خورده است. ارزش ریال هم در هفته‌های اخیر در برابر ارزهای خارجی سقوط کرده است. اقتصاد ایران پیش از شیوع کرونا نیز با مشکلات مزمن رکود، تورم، تحریم و فساد روبرو بود.""",
    "در ۲۲ ژوئن ۱۹۴۱ نیروهای محور در عملیات بارباروسا حمله سنگینی به اتحاد شوروی کرده و یکی از بزرگترین نبردهای زمینی تاریخ بشر را رقم زدند. همچنین جبهه شرقی باعث به دام افتادن نیروهای محور شد و بیش از همه ارتش آلمان نازی را درگیر جنگ فرسایشی کرد. در دسامبر ۱۹۴۱ ژاپن یک در عملیاتی ناگهانی با نام نبرد پرل هاربر به پایگاه دریایی ایالات متحده آمریکا حمله کرد. به دنبال این اتفاق آمریکا نیز بلافاصله علیه ژاپن اعلان جنگ کرد که با حمایت بریتانیا همراه شد. پس از آن متحدین (نیروهای محور در اروپا) نیز با اتحاد ژاپن علیه آمریکا اعلام جنگ کردند. دست‌آوردهای ژاپن در یورش به آمریکا باعث ایجاد این احساس در آسیا شد که آسیا از تسلط غرب خارج شده‌است از این رو بسیاری از ارتش‌های شکست خورده با آنها همراهی کردند."
]
questions = [
    "اشکالی که یک مرکز تقارن دارند",
    "غذای شترچیست؟",
    "وام یارانه به چه کسانی میدهند؟",
    "چرا امریکا وارد جنگ جهانی دوم شد؟"
]
rc_model.mt5_reading_comprehension_inference(context_list, questions, device)
```

### Evaluation
We evaluated the available models on the following datasets: 

- [**ParsiNLU - reading comprehension**](https://github.com/persiannlp/parsinlu/raw/master/data/reading_comprehension/eval.json):
               This dataset contains 570 test passage and question pairs. All these question have answers.
               This dataset is similar to Squad dataset, so we used implemented squad metrics for evaluation. 
               

All evaluation steps can be found in the [notebooks](notebooks/question-paraphrasing) associated with this task.
All the experimental results are aggregated in the corresponding [result file](evaluation-results/question_paraphrasing.xlsx).
This file contains information such as the hardware, the evaluation time, and the final results.

|  Notebook                                                                                                                  |  Model Type   |                Model Name                                 | Exact Score |  F1 Score   | 
|:--------------------------------------------------------------------------------------------------------------------------:|:-------------:|:---------------------------------------------------------:|:-----------:|------------:|
|[Link](notebooks/reading-comprehension/ReadingComprehension_persiannlp_mt5-small-parsinlu-squad-reading-comprehension.ipynb)|  mT5 (small)  | persiannlp/mt5-small-parsinlu-squad-reading-comprehension |    18.246   |   39.322    |
|[Link](notebooks/reading-comprehension/ReadingComprehension_persiannlp_mt5-base-parsinlu-squad-reading-comprehension.ipynb) |  mT5 (base)   | persiannlp/mt5-base-parsinlu-squad-reading-comprehension  |    27.018   |   54.388    |
|[Link](notebooks/reading-comprehension/ReadingComprehension_persiannlp_mt5-large-parsinlu-squad-reading-comprehension.ipynb)|  mT5 (large)  | persiannlp/mt5-large-parsinlu-squad-reading-comprehension |    36.842   |   63.930    |

## Sentiment Analysis Task
Sentiment Analysis (SA) is the study of opinions (i.e., positive, negative, or neutral sentiment) expressed in a given text, such as a review. 
Applications of SA include tasks such as market prediction, product review assessment, gauging public opinion about socio-political matters, etc.
Sentiment analysis focuses on the task of classifying a given input text by the polarity of its sentiment as being positive, negative or neutral. 
More advanced SA techniques look at whether the textual sources have associations with emotional states such as fear, anger, happiness, and sadness. 
Alternatively, instead of classifying text as being either positive, negative, or neutral, the text could be associated with a number on a pre-defined scale (e.g., -5 to +5).
Aspect-based Sentiment Analysis (ABSA) is a more fine-grained SA that aims to extract aspects of entities mentioned in the text and determine sentiment toward these aspects. 
For instance, "it tastes good but it’s so expensive ..." conveys positive and negative sentiments with respect to taste and price aspects of the mentioned product (entity), respectively.

### Sample Inference
```python
import torch
from sentiment_analysis import SentimentAnalysis

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name='m3hrdadfi/albert-fa-base-v2-sentiment-deepsentipers-binary'
sa_model = SentimentAnalysis(model_name)

texts = [
    "خوب نبود اصلا",
    "از رنگش خوشم نیومد",
    "کیفیتیش عالی بود"
]
sa_model.sentiment_analysis_inference(texts, device)
```

### Evaluation
We evaluated the available models on the following datasets: 

- **Digikala user comments**: 
            Digikala user comments provided by Open Data Mining Program (ODMP). 
            You can download this dataset from [here](https://www.kaggle.com/saeedtqp/persian-digikala-reviwes?select=2-p9vcb5bb.xlsx) 
            or [here](https://drive.google.com/file/d/1xsDSYZ_1wbEvWeFw4qE3xi83w7QPFnqQ/view?usp=sharing).
            This dataset contains 63582 user comments with three labels: no_idea, not_recommended, recommended.
            The evaluation is performed on the test set.
            In the absence of a test set, 10% of the original dataset is selected as the test set and the model is evaluated on it. 
- **SnappFood**: 
            Snappfood (an online food delivery company) user comments containing 7,000 test comments with two labels (i.e. polarity classification), Happy and Sad.
            You can download this dataset from [here](https://hooshvare.github.io/docs/datasets/sa#snappfood) 
            or [here](https://drive.google.com/uc?id=15J4zPN1BD7Q_ZIQ39VeFquwSoW8qTxgu).
- [**DeepSentiPers**](https://raw.githubusercontent.com/JoyeBright/DeepSentiPers/master/Dataset/test.csv): 
            This dataset is a balanced and augmented version of SentiPers, contains 1854 test user opinions about digital products labeled with 
            five different classes; two positives (i.e., happy and delighted), two negatives (i.e., furious and angry) and one neutral class. 
            In this database, labels are displayed using numbers: -2 (furious), -1 (angry), 0 (neutral), 1 (happy), 2 (delighted).
            This dataset can be utilized for both multi-class and binary classification. 
            In the case of binary classification, the neutral class and its corresponding sentences are removed from the dataset.
- **ParsiNLU - Sentiment Analysis**: 
            They annotated documents from food & beverages (Digikala) and movie review (Tiwall) domains.
            Their aspect-based sentiment analysis task includes three sub-tasks: 
            1) detecting the overall sentiment of a review/document, 
            2) extracting aspects toward which a sentiment is expressed, and 
            3) detecting the sentiment polarity of extracted aspects. 
            Sentiment scores are chosen from (very negative, negative, neutral, positive, very positive, mixed/borderline).
            We used [food](https://github.com/persiannlp/parsinlu/raw/master/data/sentiment-analysis/food_test.jsonl) 
            and [movie](https://github.com/persiannlp/parsinlu/raw/master/data/sentiment-analysis/movie_test.jsonl) 
            test sets for the task of Aspect-base Sentiment Analysis. 
            We also used [food](https://github.com/persiannlp/parsinlu/raw/master/data/sentiment-analysis/food.jsonl) 
            and [movie](https://github.com/persiannlp/parsinlu/raw/master/data/sentiment-analysis/movie.jsonl) 
            sets for the task of Sentiment Analysis for the reviews. 
            We don't know how to used the existing models for the task of aspect extraction, so we leave it for now. 

All evaluation steps can be found in the [notebooks](notebooks/sentiment-analysis) associated with this task.
All the experimental results are aggregated in the corresponding [result file](evaluation-results/sentiment_analysis.xlsx).
This file contains information such as the hardware, the evaluation time, and the final results.

#### Digikala
In the following table, we will report the evaluation results for 10% of `Digikala` dataset:

|  Notebook                                                                                                       |    Model Type    |             Model Name                               | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:---------------------------------------------------------------------------------------------------------------:|:----------------:|:----------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/sentiment-analysis/SentimentAnalysis_HooshvareLab_bert-fa-base-uncased-sentiment-digikala.ipynb)|   ParsBERT v2.0  | HooshvareLab/bert-fa-base-uncased-sentiment-digikala |  84.715  |        83.623	     |       78.305     |      84.715      |     76.296    |       83.969        |      76.947      |
|[Link](notebooks/sentiment-analysis/SentimentAnalysis_m3hrdadfi_albert-fa-base-v2-sentiment-digikala.ipynb)      | ALBERT-fa-base-v2| m3hrdadfi/albert-fa-base-v2-sentiment-digikala       |  81.585  |        81.082       |       74.129     |      81.585      |     73.504    |       81.314        |      73.785      |

#### SnappFood
In the following table, we will report the evaluation results for test set of `SnappFood` dataset:

|  Notebook                                                                                                        |    Model Type    |              Model Name                               | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:----------------------------------------------------------------------------------------------------------------:|:----------------:|:-----------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/sentiment-analysis/SentimentAnalysis_HooshvareLab_bert-fa-base-uncased-sentiment-snappfood.ipynb)|   ParsBERT v2.0  | HooshvareLab/bert-fa-base-uncased-sentiment-snappfood |  87.571  |       87.680        |      87.680      |      87.571      |     87.571    |        87.562       |      87.562      |
|[Link](notebooks/sentiment-analysis/SentimentAnalysis_m3hrdadfi_albert-fa-base-v2-sentiment-snappfood.ipynb)      | ALBERT-fa-base-v2| m3hrdadfi/albert-fa-base-v2-sentiment-snappfood       |  93.486  |       93.546        |      93.546      |      93.486      |     93.486    |        93.483       |      93.483      |

#### DeepSentiPers Binary
In the following table, we will report the evaluation results for test set of binary version of `DeepSentiPers` dataset:

|  Notebook                                                                                                                   |    Model Type    |                         Model Name                               | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:---------------------------------------------------------------------------------------------------------------------------:|:----------------:|:----------------------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/sentiment-analysis/SentimentAnalysis_HooshvareLab_bert-fa-base-uncased-sentiment-deepsentipers-binary.ipynb)|   ParsBERT v2.0  | HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-binary |  94.959  |       95.028        |      90.952      |      94.959      |     91.928    |       94.989        |      91.430      |
|[Link](notebooks/sentiment-analysis/SentimentAnalysis_m3hrdadfi_albert-fa-base-v2-sentiment-deepsentipers-binary.ipynb)      | ALBERT-fa-base-v2| m3hrdadfi/albert-fa-base-v2-sentiment-deepsentipers-binary       |  91.989  |       92.039        |      86.085      |      91.989      |     86.517    |       92.013        |      86.298      |
|[Link](notebooks/sentiment-analysis/SentimentAnalysis_m3hrdadfi_albert-fa-base-v2-sentiment-binary.ipynb)                    | ALBERT-fa-base-v2| m3hrdadfi/albert-fa-base-v2-sentiment-binary                     |  91.989  |       92.900        |      84.914      |      91.989      |     90.326    |       92.277        |      87.235      |

#### DeepSentiPers Multiclass
In the following tables, we will report the evaluation results for test set of `DeepSentiPers` dataset. 
These two models are able to classify each comment with one of the dataset labels: furious, angry, neutral, happy, delighted

|  Notebook                                                                                                                  |    Model Type    |                         Model Name                              | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:--------------------------------------------------------------------------------------------------------------------------:|:----------------:|:---------------------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/sentiment-analysis/SentimentAnalysis_HooshvareLab_bert-fa-base-uncased-sentiment-deepsentipers-multi.ipynb)|   ParsBERT v2.0  | HooshvareLab/bert-fa-base-uncased-sentiment-deepsentipers-multi |  71.899  |       73.961        |       62.219     |      71.899      |     70.880    |       72.202        |      65.012      |
|[Link](notebooks/sentiment-analysis/SentimentAnalysis_m3hrdadfi_albert-fa-base-v2-sentiment-deepsentipers-multi.ipynb)      | ALBERT-fa-base-v2| m3hrdadfi/albert-fa-base-v2-sentiment-deepsentipers-multi       |  68.716  |       70.125        |       52.323     |      68.716      |     56.575    |       68.669        |      53.656      |

The following model only able to classify each comment with `Negative`, `Neutral`, and `Positive` labels. 
So, we apply label conversion before evaluating this model.

|  Notebook                                                                                              |    Model Type    |                Model Name                   | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:------------------------------------------------------------------------------------------------------:|:----------------:|:-------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/sentiment-analysis/SentimentAnalysis_m3hrdadfi_albert-fa-base-v2-sentiment-multi.ipynb)| ALBERT-fa-base-v2| m3hrdadfi/albert-fa-base-v2-sentiment-multi |  67.799  |       73.256        |      66.644      |      67.799      |     62.341    |       67.602        |      62.221      |

#### Digikala+SnappFood+DeepSentiPers-Bin
For this experiment, we combine 10% of `Digikala`,  test set of `SnappFood`, and test set of `DeepSentiPers`.
Next, we convert all labels into two positive and negative labels:
- SnappFood: positive (HAPPY), negative (SAD)
- Digikala: positive (recommended), negative (not_recommended)
- DeepSentiPers: positives (happy and delighted), negatives (furious and angry)
        
|  Notebook                                                                                               |    Model Type    |                Model Name                    | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:-------------------------------------------------------------------------------------------------------:|:----------------:|:--------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/sentiment-analysis/SentimentAnalysis_m3hrdadfi_albert-fa-base-v2-sentiment-binary.ipynb)| ALBERT-fa-base-v2| m3hrdadfi/albert-fa-base-v2-sentiment-binary |  92.748  |       93.071        |       92.108     |      92.748      |     93.184    |       92.794        |      92.522      |

#### Digikala+SnappFood+DeepSentiPers-Multiclass
For this experiment, we combine 10% of `Digikala`,  test set of `SnappFood`, and test set of `DeepSentiPers`.
Next, we convert all labels into positive, negative, and neutral labels:
- SnappFood: positive (HAPPY), negative (SAD)
- Digikala: positive (recommended), negative (not_recommended), and neutral (no_idea)
- DeepSentiPers: positives (happy and delighted), negatives (furious and angry), neutral
        
|  Notebook                                                                                              |    Model Type    |                Model Name                   | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:------------------------------------------------------------------------------------------------------:|:----------------:|:-------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/sentiment-analysis/SentimentAnalysis_m3hrdadfi_albert-fa-base-v2-sentiment-multi.ipynb)| ALBERT-fa-base-v2| m3hrdadfi/albert-fa-base-v2-sentiment-multi |  73.069  |       82.996        |       70.928     |      73.069      |     77.451    |       75.596        |      69.646      |

#### Sentence Sentiment ParsiNLU - Food subset
In the following tables, we will report the evaluation results for `food` dataset:

|  Notebook                                                                                           |  Model Type  |             Model Name                           | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:---------------------------------------------------------------------------------------------------:|:------------:|:------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/sentiment-analysis/Sentiment_persiannlp_mt5-small-parsinlu-sentiment-analysis.ipynb)|  mT5 (small) | persiannlp/mt5-small-parsinlu-sentiment-analysis |  63.542  |       65.570        |       44.787     |      63.542      |     41.684    |       63.949        |      42.376      |
|[Link](notebooks/sentiment-analysis/Sentiment_persiannlp_mt5-base-parsinlu-sentiment-analysis.ipynb) |  mT5 (base)  | persiannlp/mt5-base-parsinlu-sentiment-analysis  |  63.021  |       69.641        |       49.114     |      63.021      |     42.656    |       65.411        |      44.886      |
|[Link](notebooks/sentiment-analysis/Sentiment_persiannlp_mt5-large-parsinlu-sentiment-analysis.ipynb)|  mT5 (large) | persiannlp/mt5-large-parsinlu-sentiment-analysis |  69.271  |       76.092        |       55.114     |      69.271      |     47.613    |       71.816        |      50.228      |

#### Sentence Sentiment ParsiNLU - Movie subset
In the following tables, we will report the evaluation results for `movie` dataset:

|  Notebook                                                                                           |  Model Type  |             Model Name                           | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:---------------------------------------------------------------------------------------------------:|:------------:|:------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/sentiment-analysis/Sentiment_persiannlp_mt5-small-parsinlu-sentiment-analysis.ipynb)|  mT5 (small) | persiannlp/mt5-small-parsinlu-sentiment-analysis |  54.839  |       56.277        |       41.038     |      54.839      |     39.338    |       55.240        |      39.894      |
|[Link](notebooks/sentiment-analysis/Sentiment_persiannlp_mt5-base-parsinlu-sentiment-analysis.ipynb) |  mT5 (base)  | persiannlp/mt5-base-parsinlu-sentiment-analysis  |  59.140  |       63.847        |       46.110     |      59.140      |     42.348    |       60.758        |      43.684      |
|[Link](notebooks/sentiment-analysis/Sentiment_persiannlp_mt5-large-parsinlu-sentiment-analysis.ipynb)|  mT5 (large) | persiannlp/mt5-large-parsinlu-sentiment-analysis |  62.366  |       65.785        |       47.036     |      62.366      |     44.605    |       62.074        |      44.259      |

#### Aspect Sentiment ParsiNLU - Food subset
In the following tables, we will report the evaluation results for test set of `food` dataset for the task of aspect sentiment analysis:

|  Notebook                                                                                           |  Model Type  |             Model Name                           | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:---------------------------------------------------------------------------------------------------:|:------------:|:------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/sentiment-analysis/Sentiment_persiannlp_mt5-small-parsinlu-sentiment-analysis.ipynb)|  mT5 (small) | persiannlp/mt5-small-parsinlu-sentiment-analysis |  87.426  |       86.774        |       57.840     |      87.426      |     56.966    |       87.023        |      57.098      |
|[Link](notebooks/sentiment-analysis/Sentiment_persiannlp_mt5-base-parsinlu-sentiment-analysis.ipynb) |  mT5 (base)  | persiannlp/mt5-base-parsinlu-sentiment-analysis  |  88.170  |       88.155        |       59.115     |      88.170      |     60.398    |       88.082        |      59.481      |
|[Link](notebooks/sentiment-analysis/Sentiment_persiannlp_mt5-large-parsinlu-sentiment-analysis.ipynb)|  mT5 (large) | persiannlp/mt5-large-parsinlu-sentiment-analysis |  90.327  |       90.053        |       63.685     |      90.327      |     64.622    |       90.136        |      63.935      |

#### Aspect Sentiment ParsiNLU - Movie subset
In the following tables, we will report the evaluation results for test set of `movie` dataset for the task of aspect sentiment analysis:

|  Notebook                                                                                           |  Model Type  |             Model Name                           | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:---------------------------------------------------------------------------------------------------:|:------------:|:------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/sentiment-analysis/Sentiment_persiannlp_mt5-small-parsinlu-sentiment-analysis.ipynb)|  mT5 (small) | persiannlp/mt5-small-parsinlu-sentiment-analysis |  84.926  |       84.612        |       52.455     |      84.926      |     53.480    |       84.563        |      52.474      |
|[Link](notebooks/sentiment-analysis/Sentiment_persiannlp_mt5-base-parsinlu-sentiment-analysis.ipynb) |  mT5 (base)  | persiannlp/mt5-base-parsinlu-sentiment-analysis  |  86.642  |       86.226        |       55.414     |      86.642      |     56.831    |       86.328        |      55.807      |
|[Link](notebooks/sentiment-analysis/Sentiment_persiannlp_mt5-large-parsinlu-sentiment-analysis.ipynb)|  mT5 (large) | persiannlp/mt5-large-parsinlu-sentiment-analysis |  89.583  |       89.881        |       76.223     |      89.583      |     69.413    |       89.627        |      70.907      |

## Summarization Task
Text summarization is the task of automatically generating a brief summary from a given text while maintaining the key information.

### Sample Inference
```python
import torch
from summarization import Summarization

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = 'm3hrdadfi/bert2bert-fa-news-headline'
b2b_model = Summarization(model_name=model_name, model_type="bert2bert")

sequence_list = [
    "قبل از به وجود آمدن دی سی، در خلا و فضایی عاری از هرگونه حیات که تنها پرایمال مانیتور بود، یهوه بوسیله قدرت های نامحدود دو برادر خلق کرد؛ یکی از آن ها میکائیل دمیورگوس، و دیگری سمائیل نام گرفت که بعدها با عنوان لوسیفر مورنینگ استار شناخته شد. پس از شکل گیری این دو تن، یهوه آن ها را هدایت نمود و به آن ها چگونگی استفاده از قدرت هایشان را آموخت، در نتیجه آن ها شکلی از خلقت را ایجاد کردند که هم اکنون به عنوان فرضیه چندجهانی دی سی شناخته می شود. میلیاردها سال پیش، لوسیفر فرشته مقرب دست به شورشی علیه پادشاهی بهشت زد و در نتیجه به فضایی عاری از ماده و فاقد هر گونه شکل تحت عنوان چائوپلازم تبعید شد. سپس چائوپلازم تبدیل بهک فضای متروک، ویران و گستره ای تهی با عنوان دوزخ شد، مقصد نهایی برای ارواح ملعون، جایی که مورنینگ استار فرمانروایی می کرد و در انتظار روزی بود تا بتواند دوباره آزاد شود. زمانی که تاریکی اعظم (شیطان وحشی بزرگ) بیدار شده و بازگشت، لوسیفر مجبور شد قدرت خود را با او سهیم شود و فرمانروایی خود را با بعل الذباب و عزازیل به اشتراک گذاشت. بدین سبب سه قدرت مثلثی شکل گرفتند، اما با این حال لوسیفر بخش کثیر قدرت را برای خود نگاه داشت. زمانی فرار رسید که دیریم یکی از اندلس برای جستجوی سکان خود که از او به سرقت رفته بود وارد دوزخ شد. دیریم پس از ورود به جهنم در یک نبرد ذهنی با یک دیو خبیث قدرتمند شرکت کرد و خواستار سکان دزدیده شده خود بود. دیریم پس از اینکه سکان خود را بازیافت لوسیفر را در مقابل تمام شیاطین دوزخ تحقیر کرد، و مورنینگ استار در آن روز سوگند به نابودی دیریم نمود"
]
b2b_model.bert2bert_summarization_inference(sequence_list, device, max_length=512)
```

### Evaluation
We evaluated the available models on the following datasets: 
- **Wiki Summary v1.0.0**([Link](https://github.com/m3hrdadfi/wiki-summary/tree/master/datasets/wiki_summary_persian): 
            Wiki Summary is a summarization dataset extracted from Persian Wikipedia into the form of articles and highlights.
            The dataset extracted from Persian Wikipedia into the form of articles and highlights.
            They cleaned the dataset into pairs of articles and highlights and reduced the articles' length (only version 1.0.0) 
            and highlights' length to a maximum of 512 and 128, respectively, suitable for parsBERT. 
            There are 5637 test article along with their highlights.
- **Wiki Summary v2.0.0**([Link](https://github.com/m3hrdadfi/wiki-summary/tree/master/datasets/wiki_summary_persian): 
            Wiki Summary is a summarization dataset extracted from Persian Wikipedia into the form of articles and highlights.
            The dataset extracted from Persian Wikipedia into the form of articles and highlights.
            They cleaned the dataset into pairs of articles and highlights and reduced the articles' length (only version 1.0.0) 
            and highlights' length to a maximum of 512 and 128, respectively, suitable for parsBERT. 
            There are 3754 test article along with their highlights.
- **VoA Persian Corpus v1.0.0**([Link](https://github.com/m3hrdadfi/news-headline-generation/tree/master/datasets/news_headline): 
            The dataset includes the Persian news between 2003-2008 from VoA News Network. In this particular example, 
            they cleaned the dataset into pairs of articles and headlines and reduced the articles' length to a maximum of 512, suitable for parsBERT.    

All evaluation steps can be found in the [notebooks](notebooks/summarization) associated with this task.
All the experimental results are aggregated in the corresponding [result file](evaluation-results/summarization.xlsx).
This file contains information such as the hardware, the evaluation time, and the final results.

#### Wiki Summary v1.0.0 (mid values)
The following table summarizes the ROUGE scores obtained by the existing summarization models:
Due to space constraints, only mid scores are provided in this table.

|  Notebook                                                                              | Model Type |   Model Name                         | ROUGE-1 - precision | ROUGE-1 - recall | ROUGE-1 - fmeasure | ROUGE-2 - precision | ROUGE-2 - recall | ROUGE-2 - fmeasure | ROUGE-L - precision | ROUGE-L - recall | ROUGE-L - fmeasure | ROUGE-Lsum - precision | ROUGE-Lsum - recall | ROUGE-Lsum - fmeasure |  
|:--------------------------------------------------------------------------------------:|:----------:|:------------------------------------:|:-------------------:|:----------------:|:------------------:|:-------------------:|:----------------:|:------------------:|:-------------------:|:----------------:|:------------------:|:----------------------:|:-------------------:|:---------------------:|
|[Link](notebooks/summarization/Summarization_m3hrdadfi_bert2bert-fa-news-headline.ipynb)|  bert2bert | m3hrdadfi/bert2bert-fa-news-headline |       28.106        |       4.253      |        6.993       |        3.815        |      0.526       |        0.874       |       24.482        |      3.689       |       6.062        |         24.478         |        3.690        |         6.061         |
|[Link](notebooks/summarization/Summarization_m3hrdadfi_bert2bert-fa-wiki-summary.ipynb) |  bert2bert | m3hrdadfi/bert2bert-fa-wiki-summary  |       27.144        |      28.439      |       25.832       |        6.693        |      7.576       |        6.536       |       18.605        |     20.458       |      18.021        |         18.605         |       20.457        |        18.019         |

#### Wiki Summary v2.0.0 (mid values)
The following table summarizes the ROUGE scores obtained by the existing summarization models:
Due to space constraints, only mid scores are provided in this table.

|  Notebook                                                                              | Model Type |   Model Name                         | ROUGE-1 - precision | ROUGE-1 - recall | ROUGE-1 - fmeasure | ROUGE-2 - precision | ROUGE-2 - recall | ROUGE-2 - fmeasure | ROUGE-L - precision | ROUGE-L - recall | ROUGE-L - fmeasure | ROUGE-Lsum - precision | ROUGE-Lsum - recall | ROUGE-Lsum - fmeasure |  
|:--------------------------------------------------------------------------------------:|:----------:|:------------------------------------:|:-------------------:|:----------------:|:------------------:|:-------------------:|:----------------:|:------------------:|:-------------------:|:----------------:|:------------------:|:----------------------:|:-------------------:|:---------------------:|
|[Link](notebooks/summarization/Summarization_m3hrdadfi_bert2bert-fa-news-headline.ipynb)|  bert2bert | m3hrdadfi/bert2bert-fa-news-headline |       33.122        |      4.269       |        7.346       |        5.106        |      0.601       |        1.043       |       28.374        |      3.605       |        6.215       |         28.389         |        3.607        |         6.217         |
|[Link](notebooks/summarization/Summarization_m3hrdadfi_bert2bert-fa-wiki-summary.ipynb) |  bert2bert | m3hrdadfi/bert2bert-fa-wiki-summary  |       30.895        |     27.316       |       27.809       |        8.052        |      7.349       |        7.342       |       20.401        |     18.424       |       18.529       |         20.404         |       18.426        |        18.534         |

#### VoA Persian Corpus v1.0.0 (mid values)
The following table summarizes the ROUGE scores obtained by the existing summarization models:
Due to space constraints, only mid scores are provided in this table.

|  Notebook                                                                              | Model Type |   Model Name                         | ROUGE-1 - precision | ROUGE-1 - recall | ROUGE-1 - fmeasure | ROUGE-2 - precision | ROUGE-2 - recall | ROUGE-2 - fmeasure | ROUGE-L - precision | ROUGE-L - recall | ROUGE-L - fmeasure | ROUGE-Lsum - precision | ROUGE-Lsum - recall | ROUGE-Lsum - fmeasure |  
|:--------------------------------------------------------------------------------------:|:----------:|:------------------------------------:|:-------------------:|:----------------:|:------------------:|:-------------------:|:----------------:|:------------------:|:-------------------:|:----------------:|:------------------:|:----------------------:|:-------------------:|:---------------------:|
|[Link](notebooks/summarization/Summarization_m3hrdadfi_bert2bert-fa-news-headline.ipynb)|  bert2bert | m3hrdadfi/bert2bert-fa-news-headline |       40.830        |     38.962       |       38.997       |       22.256        |     20.966       |       21.089       |       37.548        |      35.822      |       35.870       |         37.566         |       35.808        |        35.872         |
|[Link](notebooks/summarization/Summarization_m3hrdadfi_bert2bert-fa-wiki-summary.ipynb) |  bert2bert | m3hrdadfi/bert2bert-fa-wiki-summary  |        5.002        |     28.278       |        8.401       |        1.050        |      6.653       |        1.789       |        4.299        |      24.588      |        7.232       |          4.300         |       24.572        |         7.232         |

## Text Classification Task
Text classification (a.k.a. text categorization or text tagging) is the task of assigning a set of predefined categories to open-ended text.

### Sample Inference
```python
import torch
from text_classification import TextClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name='HooshvareLab/bert-fa-base-uncased-clf-persiannews'
tc_model = TextClassifier(model_name)

test_samples = [
    'حسن جوهرچی بازیگر سینما و تلویزیون ایران در گفتگو با خبرنگار حوزه سینما گروه فرهنگی باشگاه خبرنگاران جوان؛ در خصوص علت کم کاری\u200cاش در چند سال اخیر گفت: با در نظر گرفتن نبود بودجه کافی که در پی آن تولید کم خواهد شد اکثر بازیگران کم کار می\u200cشوند امیدوارم وضعیت بودجه رو به بهبود رود و تولیدات مثل قدیم افزایش یابد تا اینکه حرکت جدیدی اتفاق بیفتد و ما دوباره به طور دائم سرکار باشیم. وی در خصوص حال و هوای این روزهای سینما ایران بیان کرد: به نظر می\u200cرسد که سینما کم\u200cکم در حال تعطیل شدن است، یعنی آثار سینمایی ما مخاطب را جذب نمی\u200cکند، سالن\u200cها خالی و فیلمسازان خوب با کم کاری و یا اصلا فعالیت نمی\u200cکنند که این جای تاسف دارد، امیدوارم مسئولان سینمایی فکری به حال این موضوع به خصوص در بخش تولید و فیلم\u200cنامه داشته باشند تا اینکه سینما به روزهای درخشان خود بازگردد. وی ادامه داد: بحث فیلم\u200cنامه اولین موضوع در تولید و ساختار یک فیلم است و سینمای ما از قدیم\u200cالایام با این جریان مشکل داشته در صورتی که اواخر دهه شصت و اوایل دهه هفتاد آثار خوبی در سینما ساخته می\u200cشد اما متاسفانه این حرکت ادامه پیدا نکرد. جوهرچی بیان کرد: در دهه\u200cهای مذکور فیلم\u200cهای خوبی مانند ناخدا خورشید، اجاره\u200cنشین\u200cها و … ساختیم که شاید دلیل آن این بود که فیلم\u200cنامه نویسان ما با فراغ بال بهتری کار می\u200cکردند اما بعدها ممیزی\u200cهای گوناگون روی بحث فیلم\u200cنامه صورت گرفت و برخی خودسانسوری\u200c را شروع کردند و در کنار این موضوع تولید در تلویزیون زیاد شد و در آن سال\u200cها فیلم\u200cنامه\u200cها خوبی به تلویزیون راه پیدا کرد این شد مکه دست سینما بسته شد یعنی از نیمه دوم دهه هفتاد سینمای ما سیر نزولی را طی می\u200cکند. جوهرچی اظهار داشت: سیر نزولی از نیمه دوم دهه هفتاد تا به امروز با وجود اینکه فیلم\u200cهای خوبی مانند مادر، کمال\u200cالملک و … را داشتیم پیامد خوبی به همراه نخواهد داشت و باید به این موضوع مهم توجه بیشتری شود و فکری به حالش کرد. این بازیگر عنوان کرد: سینمای حال حاضر ما اثر قابل ملاحظه\u200cای را نمی\u200cسازد، سال\u200cهایی بود که از بین حدود هفتاد فیلم تولید شده حدود ۴۰ فیلم قابل دیدن بود اما متاسفانه در حال حاضر این رقم (۴۰ فیلم) به دو یا ۳ فیلم کاهش یافته و بقیه قابل دیدن نیست یعنی می\u200cتوان گفت که اکثر کارها سخیف هستند و سطح پایین و از فقدان کارگردان و بازیگر سطح بالا رنج می\u200cبرد. وی در پایان خاطرنشان کرد: امیدوارم سینما مثل سال\u200cهای قبل شکل و شمایل بهتری به خود بگیرد و دوباره شاهد پویایی و رونق هر چه بیشتر آن باشیم.', 
    'به گزارش گروه بین الملل باشگاه خبرنگاران جوان به نقل از هیل، آخرین تلاش نمایندگان دموکرات کنگره برای حذف کمیته حقیقت یاب بنغازی بی نتیجه ماند. تلاش نمایندگان دموکرات برای حذف کمیته حقیقت یاب بنغازی در حالی با شکست روبرو می\u200cشود که جمهوری\u200cخواهان کنگره، دور جدیدی از حملات انتقادی خود را علیه بی کفایتی دولت اوباما و هیلاری کلینتون – وزیر امور خارجه وقت آغاز کرده\u200cاند. با داغ شدن بحث مبارزات انتخاباتی نامزدهای انتخابات ریاست جمهوری ۲۰۱۶ آمریکا، رقبای جمهوری\u200cخواه و دموکرات می\u200cکوشند از هر فرصتی برای ضربه زدن به یکدیگر استفاده کنند. حمله به سفارت آمریکا در بنغازی در لیبی از جمله این موارد است. این حادثه، ۱۱ سپتامبر ۲۰۱۲ (۲۱ شهریور ۹۱) اتفاق افتاد. افراد مسلح با حمله به کنسولگری آمریکا، سفیر این کشور را به همراه ۴ دیپلمات و کارمند دیگر به قتل رساندند. از آن زمان، جمهوری\u200cخواهان با توسل به این حادثه کوشیده\u200cاند به دموکرات\u200cها ضربات سهمگینی وارد کنند. با داغ شدن بحث انتخابات ریاست جمهوری ۲۰۱۶ آمریکا، به نظر می\u200cرسد حادثه بنغازی بار دیگر بر سر زبان\u200cها بیافتد. تشکیل کمیته حقیقت یاب بنغازی در مجلس نمایندگان آمریکا حکایت از این موضوع دارد. حادثه بنغازی در مناظره\u200cهای نامزدهای انتخابات ریاست جمهوری آمریکا به ویژه در بحث سیاست خارجی می\u200cتواند تعیین کننده باشد. هیلاری کلینتون – نامزد دموکرات انتخابات ریاست جمهوری ۲۰۱۶ در زمان وقوع این حادثه وزیر امور خارجه آمریکا بود. جمهوری\u200cخواهان می\u200cکوشند با توسل به این حادثه، کفایت کلینتون را زیر سوال ببرند.', 
    'به گزارش خبرنگار فوتبال و فوتسال گروه ورزشی باشگاه خبرنگاران جوان، دیدار دو تیم فوتبال استقلال و ذوب آهن در چارچوب هفته بیست و دوم رقابت\u200cهای لیگ برتر از ساعت ۱۵ آغاز می\u200cشود که حاشیه\u200cهای آن به قرار زیر است. درب\u200cهای ورزشگاه فولادشهر ساعت ۱۲ در فاصله ۳ ساعت تا آغاز بازی به روی هواداران گشوده شد. چمن فولادشهر برای برگزاری بازی امروز وضعیت ایده آلی دارد. کاوران تیم فوتبال استقلال تهران ساعت ۱۳:۲۰دقایقی قبل در میان استقبال شدید هواداران خودی هتل اسمان اصفهان را به مقصد ورزشگاه فولادشهر ترک کرد. آبی پوشان تهرانی در حلقه هواداران محاصره شده بودند و به سختى از هتل خارج شدند. در حالی که حدود ۴۰ دقیقه دیگر بازی دو تیم ذوب آهن و استقلال آغاز می\u200cشود طرفداران تیم ابی پوش جایگاه مختص خود را پر کرده\u200cاند و در حال حاضر از ورود استقلالى\u200cها به ورزشگاه جلوگیرى مى شود این اتفاق در حالی رخ داده که فقط حدود ۲۰۰ نفر برای تشویق تیم ذوب آهن تا به این لحظه در فولادشهر حاضر شده\u200cاند. هواداران تیم فوتبال استقلال با شعار استادیوم خالیه، اس اسی جا نداره به این وضعیت گلایه کرد. سید حسین حسینی دروازه بان استقلال، رضایى و اسماعیلى به شدت مورد توجه هواداران قرار گرفتندبنرهای زیادی در استادیوم نصب شده است از جمله: ذوب آهن محبوب فشارک؛ همچون کوه آهن هستی ذوب آهن؛ استقلال محبوب قلبها. چند نفر از بازیکنان تیم\u200cهای پایه ذوب آهن در پایین جایگاه ویژه حضور دارند. نام قاسم حدادی فر پس از ۸ ماه در فهرست ۱۸ نفره ذوب آهن قرار گرفت.', 
    'به\u200c گزارش گروه اقتصادی باشگاه خبرنگاران به نقل از پایگاه اطلاع\u200cرسانی وزارت نیرو (پاون)، شرکت مدیریت منابع آب ایران اعلام کرد: حجم جریان\u200cهای سطحی کشور از ابتدای سال آبی (مهر ماه ۹۳) تا پایان فروردین ماه نسبت به متوسط درازمدت در همه حوضه\u200cهای اصلی آبریز کشور ۵۴ درصد کاهش یافته است. بر اساس این گزارش، حجم روان\u200cآب\u200cهای کشور در پایان فروردین ماه سال جاری در مقایسه با مدت مشابه درازمدت در حوضه\u200cهای دریای خزر، خلیج فارس، دریاچه ارومیه، مرکزی، هامون و سرخس به ترتیب ۵۶، ۵۵، ۶۳، ۴۰، ۷۹ و ۵۵ درصد کاهش نشان می\u200cدهد. بیشترین میزان کاهش روان\u200cآب در سال جاری نسبت به دراز مدت ۴۷ ساله در حوضه هامون و مرزی شرق بوده که ۷۹ درصد کاهش را نشان می\u200cدهد. براساس این گزارش، حجم جریان\u200cهای سطحی در پایان فروردین\u200cماه سال جاری رتبه چهل و پنجم را در ۴۷ سال گذشته به خود اختصاص داده است. حجم کل روان آب\u200cهای سطحی در این زمان در دوره بلندمدت ۴۷ ساله حدود ۵۲ میلیارد و ۴۰۱ میلیون مترمکعب بوده است. بر این اساس، حجم روان\u200cآب\u200cهای کشور در پایان فروردین\u200cماه سال جاری در مقایسه با زمان مشابه سال گذشته ۵ درصد کاهش یافته و به ۲۴ میلیارد و ۲۳۴ میلیون مترمکعب رسیده است. این میزان در مدت مشابه سال گذشته ۲۵ میلیارد و ۴۶۵ میلیون مترمکعب گزارش شده بود. حجم جریان\u200cهای سطحی در فروردین\u200cماه امسال در حوضه\u200cهای دریای خزر، خلیج فارس، دریاچه ارومیه، مرکزی، هامون و سرخس به ترتیب ۳ میلیارد و ۹۴۹ میلیون مترمکعب، ۱۳ میلیارد و ۷۱۲ میلیون مترمکعب، ۹۸۶ میلیون مترمکعب، ۴ میلیارد و ۵۹۴ میلیون مترمکعب، ۲۰۵ میلیون مترمکعب و ۷۸۷ میلیون مترمکعب ثبت شده است. همچنین حجم روان\u200cآب\u200cهای کشور در فروردین\u200cماه سال جاری در مقایسه با سال گذشته در حوضه\u200cهای دریای خزر، مرکزی، سرخس به ترتیب ۱۲، ۳۵ و ۹ درصد افزایش و در حوضه\u200cهای خلیج فارس، هامون و دریاچه ارومیه به ترتیب ۱۶، ۱۴ و ۱۷ درصد کاهش یافته است.', 
    'به گزارش خبرنگار حوزه قرآن و عترت گروه فرهنگی باشگاه خبرنگاران جوان؛ ادعیه متعددی در جهت رفع بلا به ما رسیده\u200cاند که از گذشته تا به امروز اطمینان بخش قلب مؤمنان بوده\u200cاند. ما نیز در اینجا بخشی از ادعیه مرتبط با رفع بلا را آورده\u200cایم. ختم اسماء شمشیر مولا امیرالمومنین علی علیه السلام: جناب سلمان می\u200cفرماید: روی شمشیر مولا علی (ع) اسماء وکلماتی دیدم ۱۱ کلمه دیدم که نوشته شده است هرکسی بعد از نماز صبح این ۱۱ کلمه را بگوید خودش و خانواده\u200cاش و فرزندانش در حفظ و امنیت الهی بوده و همیشه در سفر و حضر و در خواب و بیداری از بلایا محفوظ هستند- اللهم إنی أسألک یا عالما بکل خفیة یا من\u200f السماء بقدرته\u200f مبنیة یا من\u200f الأرض\u200f بقدرته\u200f مدحیة یا من\u200f الشمس\u200f و القمر بنور جلاله مضیئة یا من البحار بقدرته مجریة یا منجی یوسف من رق العبودیة یا من یصرف کل نقمة و بلیة یا من حوائج السائلین عنده مقضیة یا من لیس له حاجب یغشى و لا وزیر یرشى صل على محمد و آل محمد و احفظنی فی سفری و حضری و لیلی و نهاری و یقظتی و منامی و نفسی و أهلی و مالی و ولدیو الحمد لله وحده- بحارالانوار ج ۸۳ ص ۱۹۲ – مستدرک الوسائل ج ۵ ص ۹۰ همچنین در روایتی دیگر چنین میخوانیم: تسبیحات امیرالمومنین: کیفیت این تسبیحات؛ بار سبحان الله۱۰ بار الحمدلله۱۰ بار الله اکبر۱۰ بار لا اله الا الله ۱۰ بعد از هر نماز فضیلت این تسبیحات: امیرالمؤمنین امام علی علیه السلام به براء بن عازب فرمودند: آیا کاری به تو یاد دهم که چون آن را انجام دهی، از اولیای خدا خواهی بود؟ فرمود: بله- حضرت تسبیحات فوق را به او آموزش دادند و سپس فرمودند: هر کس این تسبیحات را بعد از هر نماز بخواند، خدا هزار بلای دنیوی را از او دور می\u200cکند، که آسان\u200cترین آن بازگشت از دین است و در آخرت برای او هزارمنزلگاه آماده می\u200cکند که یکی از آن منزلت\u200cها مجاورت رسول خدا صلی الله علیه و آله است- الدعوات (سلوة الحزین) قطب الدین راوندی، ص ۴۹ مستدرک الوسائل، ج٥، ص ٨٢ بحار الانوار، چ بیروت، ج ٨٣، ص٣٤- همچنین با توجه در آیات ۲۰ تا۲۲ سوره بروج درمی\u200cیابیم هر کس با ایمان در وقت رفتن به سفر آیات زیر را در خانه خود بانیت خالص و توجه به خدا و معنی آن بنویسدو سه بار بخواند اهل و عیال و مال او همه صحیح و سالم از بلیات مانند تا آنگاه که مراجعت کند - والله من ورائهم محیط بل هو قرآن مجیدفی لوح محفوظ خواص آیات قرآن کریم ص ۲۰۱ در روایتی از حضرت صادق علیه السلام نیز چنین آمده است که هر کسی که در صبح سه مرتبه این دعا را بخواند تا شام به او بلایی نرسد و اگر در شام بگوید تا صبح به او بلایی نرسد- بسم الله الذی لا یضر مع اسمه شی\u200fء فی الأرض و لا فی السماء و هو السمیع العلیم بحارالانوار ج ۸۳ ص ۲۹۸ این روایت نیز به حضرت امام رضا علیه السلام منسوب است: هر گاه خواستى کالاى خود را در حفظ و حراست بدارى، آیة الکرسى بخوان وبنویس و آن را در وسط کالا قرار بده و نیز بنویس: وجعلنا من بین أیدیهم سدا ومن خلفهم سدا فأغشیناهم فهم \u200fلایبصرون (۱)، لا ضیعة على ما حفظه الله فإن تولوا فقل حسبی الله لاإله إلا هو، علیه توکلت، وهو رب العرش العظیم (۲) - ما پیش روى آنان سدى و پشت سرشان سدى نهاده \u200fایم و دیدگانشان را پوشانیده\u200f ایم که نمى\u200f بینند آنچه را خدا حفظ کند تباه نمى\u200f گردد پس اگر روى گرداندند خدا مرا بس است، هیچ معبودى جز او نیست، بر او توکل کردم، و او پروردگار عرش باعظمت است - (پس اگر چنین کنى) آن را در حفظ و حراست قرار داده \u200fاى اگر خدا بخواهد و به \u200fآن به اذن و فرمان پروردگار بدى نرسد- (۳) سوره یس، آیه ۹- ۱ سوره توبه، آیه ۱۲۹- ۲ بحار الأنوار:، فقه الرضا علیه السلام: ۴۰۰- ۳'
]
tc_model.text_classification_inference(test_samples) 
```

### Evaluation
We evaluated the available models on the following datasets: 
- [**DigiMAG**](https://drive.google.com/file/d/1YgrCYY-Z0h2z0-PfWVfOGt1Tv0JDI-qz): 
            A total of 8,515 articles scraped from Digikala Online Magazine. 
            This dataset includes seven different classes Video Games, Shopping Guide, Health Beauty, Science Technology, General, Art Cinema, and Books Literature.
            We apply our evaluation on test set of DigiMAG which contains 852 test article.
- [**Persian News**](https://drive.google.com/uc?id=1B6xotfXCcW9xS1mYSBQos7OCg0ratzKC): 
            A dataset of various news articles scraped from different online news agencies’ websites. 
            The total number of articles is 16,438, spread over eight different classes, Economic, International, Political, Science Technology, Cultural Art, Sport, and Medical.
            We apply our evaluation on test set of DigiMAG which contains 1644 test article.

All evaluation steps can be found in the [notebooks](notebooks/text-classification) associated with this task.
All the experimental results are aggregated in the corresponding [result file](evaluation-results/text_classification.xlsx).
This file contains information such as the hardware, the evaluation time, and the final results.

#### DigiMAG
In the following tables, we will report the evaluation results for `DigiMAG` dataset:

|  Notebook                                                                                                   |    Model Type    |              Model Name                       | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:-----------------------------------------------------------------------------------------------------------:|:----------------:|:---------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/text-classification/TextClassification_parsbert_v2.0_bert-fa-base-uncased-clf-digimag.ipynb)|   ParsBERT v2.0  | HooshvareLab/bert-fa-base-uncased-clf-digimag |  95.657  |        95.317       |       85.204     |      95.657      |     80.474    |       95.365        |      81.852      |
|[Link](notebooks/text-classification/TextClassification_albert_v2.0_albert-fa-base-v2-clf-digimag.ipynb)     | ALBERT-fa-base-v2| m3hrdadfi/albert-fa-base-v2-clf-digimag       |  95.070  |        93.776       |       77.990     |      95.070      |     75.242    |       94.303        |      75.690      |

#### Persian News
In the following tables, we will report the evaluation results for `Persian News` dataset:

|  Notebook                                                                                                       |    Model Type    |              Model Name                           | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:---------------------------------------------------------------------------------------------------------------:|:----------------:|:-------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/text-classification/TextClassification_parsbert_v2.0_bert-fa-base-uncased-clf-persiannews.ipynb)|   ParsBERT v2.0  | HooshvareLab/bert-fa-base-uncased-clf-persiannews |  98.723  |       98.724        |       98.789     |      98.723      |     98.677    |       98.722        |      98.731      |
|[Link](notebooks/text-classification/TextClassification_albert_v2.0_albert-fa-base-v2-clf-persiannews.ipynb)     | ALBERT-fa-base-v2| m3hrdadfi/albert-fa-base-v2-clf-persiannews       |  98.358  |       98.364        |       98.477     |      98.358      |     98.332    |       98.356        |      98.400      |

## Textual Entailment Task
Textual Entailment is the task of deciding whether a whether two given questions are paraphrases of each other or not.
Textual Entailment(TE) and its newer variant, Natural Language Inference(NLI), are typically defined as a 3-way classification task where the goal is to determine whether a hypothesis sentence entails, contradicts, or is neutral with respect to a given premise sentence.

### Sample Inference
```python
import torch
from textual_entailment import TextualEntailment

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name='persiannlp/mt5-small-parsinlu-snli-entailment'
te_model = TextualEntailment(model_name=model_name, model_type="mt5", label_list = ['e', 'c', 'n'])

premise_list = [
    "این مسابقات بین آوریل و دسامبر در هیپودروم ولیفندی در نزدیکی باکرکی ، ۱۵ کیلومتری (۹ مایل) غرب استانبول برگزار می شود.",
    "آیا کودکانی وجود دارند که نیاز به سرگرمی دارند؟",
    "ما به سفرهایی رفته ایم که در نهرهایی شنا کرده ایم"
]
hypothesis_list = [
    "در ولیفندی هیپودروم، مسابقاتی از آوریل تا دسامبر وجود دارد.",
    "هیچ کودکی هرگز نمی خواهد سرگرم شود.",
    "علاوه بر استحمام در نهرها ، ما به اسپا ها و سونا ها نیز رفته ایم."
]
te_model.mt5_textual_entailment_inference(premise_list, hypothesis_list, device)
```

### Evaluation
We evaluated the available models on the following datasets: 

- [**FarsTail**](https://github.com/dml-qom/FarsTail/raw/master/data/Test-word.csv): 
       This is the first relatively large-scale Persian dataset for NLI task, called FarsTail. 
       A total of 10,367 samples are generated from a collection of 3,539 multiple-choice questions. 
       The train, validation, and test portions include 7,266, 1,537, and 1,564 instances, respectively. 
       We apply our evaluation on test set of FarsTail.
- **ParsiNLU - Entailment**: 
       We construct two subsets: 
       (i) 850 pairs based on available [natural sentences](https://github.com/persiannlp/parsinlu/raw/master/data/entailment/merged_with_farstail/test_natural.tsv), and 
       (ii) 823 pairs based on available [English query-paraphrasing dataset](https://github.com/persiannlp/parsinlu/raw/master/data/entailment/merged_with_farstail/test_translation.tsv). 
       The former approach yields high quality instances, however, it is a relatively slower annotation task. 
       The latter is slightly easier, but yields less interesting instances.
       There is also a farstail set along these two dataset in [parsinlu repository](https://github.com/persiannlp/parsinlu/raw/master/data/entailment/merged_with_farstail/test_farstail.tsv).

All evaluation steps can be found in the [notebooks](notebooks/textual-entailment) associated with this task.
All the experimental results are aggregated in the corresponding [result file](evaluation-results/textual_entailment.xlsx).
This file contains information such as the hardware, the evaluation time, and the final results.

#### natural from ParsiNLU
In the following table, we will report evaluation results for `natural` subset of ParsiNLU - Entailment dataset:

|  Notebook                                                                                                      |  Model Type           |             Model Name                              | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:--------------------------------------------------------------------------------------------------------------:|:---------------------:|:---------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_mt5-small-parsinlu-snli-entailment.ipynb)      |  mT5 (small)          | persiannlp/mt5-small-parsinlu-snli-entailment       |  53.059  |       52.320        |       52.223     |      53.059      |     52.900    |       51.434        |      51.233      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_mt5-base-parsinlu-snli-entailment.ipynb)       |  mT5 (base)           | persiannlp/mt5-base-parsinlu-snli-entailment        |  57.765  |       57.321        |       57.191     |      57.765      |     57.587    |       57.234        |      57.062      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_mt5-large-parsinlu-snli-entailment.ipynb)      |  mT5 (large)          | persiannlp/mt5-large-parsinlu-snli-entailment       |  71.294  |       71.699        |       71.493     |      71.294      |     71.510    |       71.199        |      71.209      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_mbert-base-parsinlu-entailment-2.ipynb)        |  mBERT (base)         | persiannlp/mbert-base-parsinlu-entailment           |  54.235  |       52.933        |       52.785     |      54.235      |     54.095    |       53.123        |      52.954      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_parsbert-base-parsinlu-entailment-2.ipynb)     |  ParsBERT (base)      | persiannlp/parsbert-base-parsinlu-entailment        |  53.529  |       52.635        |       52.516     |      53.529      |     53.574    |       52.865        |      52.823      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_wikibert-base-parsinlu-entailment-2.ipynb)     |  WikiBERT (base)      | persiannlp/wikibert-base-parsinlu-entailment        |  54.118  |       53.245        |       53.003     |      54.118      |     54.038    |       53.494        |      53.329      |
|[Link](notebooks/textual-entailment/TextualEntailment_m3hrdadfi_bert-fa-base-uncased-farstail-2.ipynb)          |  Sentence-Transformer | m3hrdadfi/bert-fa-base-uncased-farstail             |  51.647  |       51.590        |       51.333     |      51.647      |     52.419    |       49.026        |      49.340      |
|[Link](notebooks/textual-entailment/TextualEntailment_m3hrdadfi_bert-fa-base-uncased-farstail-mean-tokens.ipynb)|  Sentence-Transformer | m3hrdadfi/bert-fa-base-uncased-farstail-mean-tokens |  40.824  |       36.309        |       35.750     |      40.824      |     37.510    |       31.860        |      30.010      |

#### mnli from ParsiNLU
In the following table, we will report evaluation results for `mnli` subset of ParsiNLU - Entailment dataset:

|  Notebook                                                                                                      |  Model Type           |             Model Name                              | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:--------------------------------------------------------------------------------------------------------------:|:---------------------:|:---------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_mt5-small-parsinlu-snli-entailment.ipynb)      |  mT5 (small)          | persiannlp/mt5-small-parsinlu-snli-entailment       |  56.258  |       57.398        |      56.734      |      56.258      |     57.093    |       56.191        |      56.224      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_mt5-base-parsinlu-snli-entailment.ipynb)       |  mT5 (base)           | persiannlp/mt5-base-parsinlu-snli-entailment        |  62.819  |       62.789        |      62.327      |      62.819      |     62.295    |       62.803        |      62.310      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_mt5-large-parsinlu-snli-entailment.ipynb)      |  mT5 (large)          | persiannlp/mt5-large-parsinlu-snli-entailment       |  73.026  |       73.166        |      72.427      |      73.026      |     72.445    |       73.080        |      72.419      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_mbert-base-parsinlu-entailment-2.ipynb)        |  mBERT (base)         | persiannlp/mbert-base-parsinlu-entailment           |  51.276  |       51.562        |      51.168      |      51.276      |     51.388    |       51.337        |      51.188      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_parsbert-base-parsinlu-entailment-2.ipynb)     |  ParsBERT (base)      | persiannlp/parsbert-base-parsinlu-entailment        |  54.313  |       54.723        |      54.668      |      54.313      |     53.964    |       54.374        |      54.181      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_wikibert-base-parsinlu-entailment-2.ipynb)     |  WikiBERT (base)      | persiannlp/wikibert-base-parsinlu-entailment        |  52.734  |       53.221        |      53.056      |      52.734      |     52.336    |       52.814        |      52.543      |
|[Link](notebooks/textual-entailment/TextualEntailment_m3hrdadfi_bert-fa-base-uncased-farstail-2.ipynb)          |  Sentence-Transformer | m3hrdadfi/bert-fa-base-uncased-farstail             |  49.939  |       56.329        |      55.485      |      49.939      |     51.533    |       49.968        |      50.267      |
|[Link](notebooks/textual-entailment/TextualEntailment_m3hrdadfi_bert-fa-base-uncased-farstail-mean-tokens.ipynb)|  Sentence-Transformer | m3hrdadfi/bert-fa-base-uncased-farstail-mean-tokens |  37.424  |       35.178        |      35.114      |      37.424      |     36.491    |       26.431        |      26.511      |

#### FarsTail from ParsiNLU
In the following table, we will report evaluation results for `Farstail` dataset inside ParsiNLU repository:

|  Notebook                                                                                                      |  Model Type           |             Model Name                              | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:--------------------------------------------------------------------------------------------------------------:|:---------------------:|:---------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_mt5-small-parsinlu-snli-entailment.ipynb)      |  mT5 (small)          | persiannlp/mt5-small-parsinlu-snli-entailment       |  76.023  |       75.864        |       75.767     |      76.023      |     75.882    |       75.926        |      75.807      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_mt5-base-parsinlu-snli-entailment.ipynb)       |  mT5 (base)           | persiannlp/mt5-base-parsinlu-snli-entailment        |  85.742  |       85.820        |       85.731     |      85.742      |     85.686    |       85.765        |      85.693      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_mt5-large-parsinlu-snli-entailment.ipynb)      |  mT5 (large)          | persiannlp/mt5-large-parsinlu-snli-entailment       |  93.414  |       93.414        |       93.400     |      93.414      |     93.398    |       93.414        |      93.399      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_mbert-base-parsinlu-entailment-2.ipynb)        |  mBERT (base)         | persiannlp/mbert-base-parsinlu-entailment           |  80.115  |       80.628        |       80.479     |      80.115      |     80.074    |       80.258        |      80.164      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_parsbert-base-parsinlu-entailment-2.ipynb)     |  ParsBERT (base)      | persiannlp/parsbert-base-parsinlu-entailment        |  79.795  |       79.956        |       79.831     |      79.795      |     79.713    |       79.854        |      79.750      |
|[Link](notebooks/textual-entailment/TextualEntailment_persiannlp_wikibert-base-parsinlu-entailment-2.ipynb)     |  WikiBERT (base)      | persiannlp/wikibert-base-parsinlu-entailment        |  82.033  |       82.542        |       82.392     |      82.033      |     81.971    |       82.137        |      82.032      |
|[Link](notebooks/textual-entailment/TextualEntailment_m3hrdadfi_bert-fa-base-uncased-farstail-2.ipynb)          |  Sentence-Transformer | m3hrdadfi/bert-fa-base-uncased-farstail             |  81.714  |       81.673        |       81.581     |      81.714      |     81.605    |       81.691        |      81.590      |
|[Link](notebooks/textual-entailment/TextualEntailment_m3hrdadfi_bert-fa-base-uncased-farstail-mean-tokens.ipynb)|  Sentence-Transformer | m3hrdadfi/bert-fa-base-uncased-farstail-mean-tokens |  38.107  |       38.765        |       28.977     |      38.107      |     28.500    |       31.577        |      23.542      |

#### FarsTail
In the following table, we will report evaluation results for `Farstail` dataset:

|  Notebook                                                                                                      |  Model Type           |             Model Name                              | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:--------------------------------------------------------------------------------------------------------------:|:---------------------:|:---------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/textual-entailment/TextualEntailment_m3hrdadfi_bert-fa-base-uncased-farstail-2.ipynb)          |  Sentence-Transformer | m3hrdadfi/bert-fa-base-uncased-farstail             |  81.650  |       81.613        |       81.520     |      81.650      |     81.541    |       81.629        |      81.528      |
|[Link](notebooks/textual-entailment/TextualEntailment_m3hrdadfi_bert-fa-base-uncased-farstail-mean-tokens.ipynb)|  Sentence-Transformer | m3hrdadfi/bert-fa-base-uncased-farstail-mean-tokens |  38.043  |       38.700        |       28.930     |      38.043      |     28.454    |       31.502        |      23.488      |

## Named Entity Recognition(NER) Task
Named-entity recognition is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.

### Sample Inference
```python
import torch
from ner import NER

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name='HooshvareLab/bert-base-parsbert-ner-uncased'
ner_model = NER(model_name)

texts = [
    "مدیرکل محیط زیست استان البرز با بیان اینکه با بیان اینکه موضوع شیرابه‌های زباله‌های انتقال یافته در منطقه حلقه دره خطری برای این استان است، گفت: در این مورد گزارشاتی در ۲۵ مرداد ۱۳۹۷ تقدیم مدیران استان شده است.",
    "به گزارش خبرگزاری تسنیم از کرج، حسین محمدی در نشست خبری مشترک با معاون خدمات شهری شهرداری کرج که با حضور مدیرعامل سازمان‌های پسماند، پارک‌ها و فضای سبز و نماینده منابع طبیعی در سالن کنفرانس شهرداری کرج برگزار شد، اظهار داشت: ۸۰٪  جمعیت استان البرز در کلانشهر کرج زندگی می‌کنند.",
    "وی افزود: با همکاری‌های مشترک بین اداره کل محیط زیست و شهرداری کرج برنامه‌های مشترکی برای حفاظت از محیط زیست در شهر کرج در دستور کار قرار گرفته که این اقدامات آثار مثبتی داشته و تاکنون نزدیک به ۱۰۰ میلیارد هزینه جهت خریداری اکس-ریس صورت گرفته است.",
]
inference_output = ner_model.ner_inference(texts, device, ner_model.config.max_position_embeddings)
```

### Evaluation
We evaluated the available models on the following datasets: 

- [**Peyma**](https://drive.google.com/file/d/1WZxpFRtEs5HZWyWQ2Pyg9CCuIBs1Kmvx/view): 
        A medium size NER corpus with 7 classes of named entities (person, location and organization, money, percent, dates, and time). 
        This corpus contains more than 700 news documents. The PEYMA dataset includes 7,145 sentences with 302,530 tokens 
        from which 41,148 tokens are tagged in IOB format in with seven different classes, Organization, Percent, Money, 
        Location, Date, Time, and Person. It has been prepared by Iran Telecommunication Research Center (ITRC). 
        The training data files contain two columns separated by a tab. Each word has been put on a separate line and 
        there is an empty line after each sentence. The first item on each line is a word, and the second named entity tag. 
        The named entity tags have the format I-TYPE which means that the word is inside a phrase of type TYPE. The 
        first word of each named entity have tag B-TYPE to show that it starts a new named entity. A word with tag O is 
        not part of a named entity. 
        We will evaluate existing ner models using test set of Peyma which includes 1026 sentences.
- **Arman**: 
        This is the first manually-annotated Persian named-entity (NE) dataset. 
        ARMAN dataset holds 7,682 sentences with 250,015 sentences tagged over six different classes: person, organization 
        (such as banks, ministries, embassies, teams, nationalities, networks and publishers), location (such as cities, 
        villages, rivers, seas, gulfs, deserts and mountains), facility (such as schools, universities, research centers, 
        airports, railways, bridges, roads, harbors, stations, hospitals, parks, zoos and cinemas), product (such as books, 
        newspapers, TV shows, movies, airplanes, ships, cars, theories, laws, agreements and religions), and event (such as 
        wars, earthquakes, national holidays, festivals and conferences).
        It is available in 3 folds to be used in turn as training and test sets. 
        Each file contains one token, along with its manually annotated named-entity tag, per line. 
        Each sentence is separated with a newline. The NER tags are in IOB format. 
        We will evaluate existing ner models using test set of Arman which includes 7681 sentences.
- [**WikiAnn**](https://drive.google.com/file/d/1QOG15HU8VfZvJUNKos024xI-OGm0zhEX/view?usp=sharing): 
        This dataset is one of the datasets available in [link](https://elisa-ie.github.io/wikiann/).
        This work is about a cross-lingual name tagging and linking framework for languages that exist in Wikipedia. 
        Given a document in any of these languages, this framework is able to identify name mentions, assign a 
        coarse-grained or fine-grained type to each mention, and link it to an English Knowledge Base if it is linkable.
        We will evaluate existing ner models using persian subset of this dataset which contains 272266 sentences.
        In the absence of a test set, we evaluate our models using whole `WikiAnn` dataset.
- **Arman+Peyma+WikiAnn**:
        A mixed NER dataset collected from ARMAN, PEYMA, and WikiANN that covered ten types of entities: Date (DAT), 
        Event (EVE), Facility (FAC), Location (LOC), Money (MON), Organization (ORG), Percent (PCT), Person (PER), 
        Product (PRO), Time (TIM).
        You can download this dataset from [here](https://github.com/hooshvare/parsner) 
        or [here](https://drive.google.com/uc?id=1fC2WGlpqumUTaT9Dr_U1jO2no3YMKFJ4).

All evaluation steps can be found in the [notebooks](notebooks/named-entity-recognition) associated with this task.
All the experimental results are aggregated in the corresponding [result file](evaluation-results/named-entity-recognition.xlsx).
This file contains information such as the hardware, the evaluation time, and the final results.

#### Peyma
The following table shows the statistics of the entities within this dataset: 

| B_ORG | I_ORG | B_LOC | I_LOC | B_DAT | I_DAT | B_PER | I_PER | B_PCT | I_PCT | B_TIM | I_TIM | B_MON | I_MON |   O   |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  667  | 1104  |  595  |  211  |  208  |  236  |  434  |  297  |  36   |  40   |  16   |  24   |  26   |  65   | 28215 |

In the following table, we will report evaluation results for test set of `Peyma` dataset:

|  Notebook                                                                                            |  Model Type       |             Model Name                           | Accuracy | Precision (weighted)| Precision (micro)| Precision (macro)| Recall (weighted)| Recall (micro)| Recall (macro)| F1-Score (weighted) | F1-Score (micro) | F1-Score (macro) |
|:----------------------------------------------------------------------------------------------------:|:-----------------:|:------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:----------------:|:-------------:|:-------------:|:-------------------:|:----------------:|:----------------:|
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-peymaner-uncased.ipynb) | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-peymaner-uncased |  98.237  |       87.297        |      87.960      |      87.460      |      77.841      |     77.841    |     74.521    |       82.106        |      82.591      |      80.114      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-armanner-uncased.ipynb) | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-armanner-uncased |  95.606  |       68.940        |      67.659      |      69.208      |      56.739      |     56.739    |     56.263    |       61.911        |      61.719      |      61.764      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-ner-uncased.ipynb)      | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-ner-uncased      |  96.610  |       79.854        |      80.958      |      70.021      |      68.553      |     68.553    |     50.508    |       72.836        |      74.240      |      57.095      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-base-uncased-ner-peyma.ipynb)      | ParsBERT v2.0     | HooshvareLab/bert-fa-base-uncased-ner-peyma      |  97.583  |       86.089        |      86.806      |      86.296      |      70.159      |     70.159    |     68.293    |       77.160        |      77.600      |      75.832      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-base-uncased-ner-arman.ipynb)      | ParsBERT v2.0     | HooshvareLab/bert-fa-base-uncased-ner-arman      |  95.137  |       66.946        |      64.388      |      67.460      |      51.758      |     51.758    |     50.947    |       57.690        |      57.387      |      57.422      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_albert-fa-zwnj-base-v2-ner.ipynb)          | ALBERT v3.0       | HooshvareLab/albert-fa-zwnj-base-v2-ner          |  57.224  |       28.033        |       6.243      |      25.267      |      13.161      |     13.161    |      9.034    |        7.623        |       8.469      |       8.369      |
|[Link](notebooks/named-entity-recognition/NER_m3hrdadfi_albert-fa-base-v2-ner-peyma.ipynb)            | ALBERT-fa-base-v2 | m3hrdadfi/albert-fa-base-v2-ner-peyma            |  90.008  |       57.606        |      58.898      |      48.904      |      15.714      |     15.714    |     11.037    |       24.199        |      24.809      |      17.547      |
|[Link](notebooks/named-entity-recognition/NER_m3hrdadfi_albert-fa-base-v2-ner-arman.ipynb)            | ALBERT-fa-base-v2 | m3hrdadfi/albert-fa-base-v2-ner-arman            |  93.238  |       56.118        |      53.779      |      56.072      |      28.130      |     28.130    |     27.014    |       36.186        |      36.938      |      35.229      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-zwnj-base-ner.ipynb)               | ParsBERT v3.0     | HooshvareLab/bert-fa-zwnj-base-ner               |  95.140  |       65.536        |      65.864      |      63.316      |      56.368      |     56.368    |     53.801    |       60.453        |      60.748      |      57.971      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_distilbert-fa-zwnj-base-ner.ipynb)         | DistilBERT v3.0   | HooshvareLab/distilbert-fa-zwnj-base-ner         |  94.716  |       64.108        |      63.857      |      67.073      |      51.913      |     51.913    |     49.726    |       57.209        |      57.269      |      56.493      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_roberta-fa-zwnj-base-ner.ipynb)            | Roberta v3.0      | HooshvareLab/roberta-fa-zwnj-base-ner            |          |                     |                  |                  |                  |               |               |                     |                  |                  |

#### Arman
The following table shows the statistics of the entities within this dataset: 

| B-org | I-org | B-loc | I-loc | B-pers | I-pers | B-event | I-event | B-pro | I-pro | B-fac | I-fac |   O    |
|:-----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:-------:|:-----:|:-----:|:-----:|:-----:|:------:|
| 4533  | 5503  | 3408  |  900  |  3275  |  1940  |   580   |  1939   |  724  |  739  |  550  |  936  | 224969 | 

In the following table, we will report evaluation results for test set of `Arman` dataset:

|  Notebook                                                                                            |  Model Type       |             Model Name                           | Accuracy | Precision (weighted)| Precision (micro)| Precision (macro)| Recall (weighted)| Recall (micro)| Recall (macro)| F1-Score (weighted) | F1-Score (micro) | F1-Score (macro) |
|:----------------------------------------------------------------------------------------------------:|:-----------------:|:------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:----------------:|:-------------:|:-------------:|:-------------------:|:----------------:|:----------------:|
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-peymaner-uncased.ipynb) | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-peymaner-uncased |  95.799  |       63.599        |      61.126      |       63.623     |      59.798      |     59.798    |     61.203    |       60.290        |      60.455      |      60.963      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-armanner-uncased.ipynb) | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-armanner-uncased |  98.358  |       86.175        |      86.388      |       83.092     |      77.055      |     77.055    |     74.742    |       81.256        |      81.455      |      78.505      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-ner-uncased.ipynb)      | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-ner-uncased      |  97.535  |       76.425        |      76.002      |       73.658     |      72.460      |     72.460    |     65.950    |       74.051        |      74.189      |      69.231      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-base-uncased-ner-peyma.ipynb)      | ParsBERT v2.0     | HooshvareLab/bert-fa-base-uncased-ner-peyma      |  95.650  |       65.333        |      62.779      |       65.320     |      51.870      |     51.870    |     53.134    |       56.480        |      56.806      |      57.162      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-base-uncased-ner-arman.ipynb)      | ParsBERT v2.0     | HooshvareLab/bert-fa-base-uncased-ner-arman      |  97.973  |       84.227        |      84.338      |       81.932     |      74.404      |     74.404    |     74.008    |       78.816        |      79.060      |      77.481      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_albert-fa-zwnj-base-v2-ner.ipynb)          | ALBERT v3.0       | HooshvareLab/albert-fa-zwnj-base-v2-ner          |  18.349  |       20.751        |       1.872      |       18.634     |       3.560      |      3.560    |      2.607    |        2.902        |       2.454      |       2.886      |
|[Link](notebooks/named-entity-recognition/NER_m3hrdadfi_albert-fa-base-v2-ner-peyma.ipynb)            | ALBERT-fa-base-v2 | m3hrdadfi/albert-fa-base-v2-ner-peyma            |  92.519  |       37.436        |      33.373      |       39.187     |       6.254      |      6.254    |      6.540    |       10.332        |      10.534      |      10.798      |
|[Link](notebooks/named-entity-recognition/NER_m3hrdadfi_albert-fa-base-v2-ner-arman.ipynb)            | ALBERT-fa-base-v2 | m3hrdadfi/albert-fa-base-v2-ner-arman            |  93.631  |       61.832        |      62.547      |       52.273     |      28.587      |     28.587    |     22.760    |       38.645        |      39.239      |      31.064      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-zwnj-base-ner.ipynb)               | ParsBERT v3.0     | HooshvareLab/bert-fa-zwnj-base-ner               |  92.640  |       54.481        |      32.944      |       45.902     |      51.236      |     51.236    |     47.538    |       50.872        |      40.102      |      41.875      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_distilbert-fa-zwnj-base-ner.ipynb)         | DistilBERT v3.0   | HooshvareLab/distilbert-fa-zwnj-base-ner         |  96.586  |       62.768        |      62.479      |       57.043     |      53.755      |     53.755    |     48.977    |       57.665        |      57.790      |      51.919      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_roberta-fa-zwnj-base-ner.ipynb)            | Roberta v3.0      | HooshvareLab/roberta-fa-zwnj-base-ner            |          |                     |                  |                  |                  |               |               |                     |                  |                  |

#### Peyma+Arman
In the following table, we will report evaluation results for combination of test sets of `Peyma` and `Arman` datasets:

|  Notebook                                                                                            |  Model Type       |             Model Name                           | Accuracy | Precision (weighted)| Precision (micro)| Precision (macro)| Recall (weighted)| Recall (micro)| Recall (macro)| F1-Score (weighted) | F1-Score (micro) | F1-Score (macro) |
|:----------------------------------------------------------------------------------------------------:|:-----------------:|:------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:----------------:|:-------------:|:-------------:|:-------------------:|:----------------:|:----------------:|
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-peymaner-uncased.ipynb) | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-peymaner-uncased |  94.315  |       65.178        |      56.610      |       34.274     |      62.497      |     62.497    |     69.367    |       62.057        |      59.408      |      37.299      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-armanner-uncased.ipynb) | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-armanner-uncased |  97.955  |       83.592        |      83.627      |       79.262     |      74.538      |     74.538    |     73.208    |       78.685        |      78.821      |      75.852      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-ner-uncased.ipynb)      | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-ner-uncased      |  97.192  |       75.767        |      75.153      |       53.586     |      71.570      |     71.570    |     50.035    |       73.282        |      73.318      |      50.362      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-base-uncased-ner-peyma.ipynb)      | ParsBERT v2.0     | HooshvareLab/bert-fa-base-uncased-ner-peyma      |  94.318  |       66.490        |      57.296      |       34.896     |      54.061      |     54.061    |     60.860    |       57.939        |      55.631      |      35.448      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-base-uncased-ner-arman.ipynb)      | ParsBERT v2.0     | HooshvareLab/bert-fa-base-uncased-ner-arman      |  97.536  |       81.489        |      81.425      |       78.055     |      71.530      |     71.530    |     72.308    |       75.944        |      76.158      |      74.658      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_albert-fa-zwnj-base-v2-ner.ipynb)          | ALBERT v3.0       | HooshvareLab/albert-fa-zwnj-base-v2-ner          |  18.736  |       20.030        |       1.887      |       11.014     |       3.556      |      3.556    |      1.560    |        2.700        |       2.466      |       1.673      |
|[Link](notebooks/named-entity-recognition/NER_m3hrdadfi_albert-fa-base-v2-ner-peyma.ipynb)            | ALBERT-fa-base-v2 | m3hrdadfi/albert-fa-base-v2-ner-peyma            |  91.902  |       38.052        |      31.491      |       18.552     |       6.526      |      6.526    |      4.279    |       10.731        |      10.811      |       6.072      |
|[Link](notebooks/named-entity-recognition/NER_m3hrdadfi_albert-fa-base-v2-ner-arman.ipynb)            | ALBERT-fa-base-v2 | m3hrdadfi/albert-fa-base-v2-ner-arman            |  93.522  |       61.240        |      61.424      |       50.900     |      28.299      |     28.299    |     22.403    |       38.197        |      38.747      |      30.426      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-zwnj-base-ner.ipynb)               | ParsBERT v3.0     | HooshvareLab/bert-fa-zwnj-base-ner               |  92.626  |       54.540        |      34.243      |       42.270     |      51.619      |     51.619    |     50.075    |       51.273        |      41.173      |      42.135      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_distilbert-fa-zwnj-base-ner.ipynb)         | DistilBERT v3.0   | HooshvareLab/distilbert-fa-zwnj-base-ner         |  96.289  |       62.214        |      61.895      |       56.616     |      53.016      |     53.016    |     45.599    |       57.009        |      57.112      |      49.725      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_roberta-fa-zwnj-base-ner.ipynb)            | Roberta v3.0      | HooshvareLab/roberta-fa-zwnj-base-ner            |          |                     |                  |                  |                  |               |               |                     |                  |                  |

#### WikiAnn
The following table shows the statistics of the entities within this dataset: 

| B-ORG  | I-ORG  | B-LOC  | I-LOC  | B-PER | I-PER  |   O    |
|:------:|:------:|:------:|:------:|:-----:|:------:|:------:|
| 106811 | 398513 | 136732 | 322138 | 50701 | 113338 | 486529 |

In the following table, we will report evaluation results for `WikiAnn` dataset:

|  Notebook                                                                                            |  Model Type       |             Model Name                           | Accuracy | Precision (weighted)| Precision (micro)| Precision (macro)| Recall (weighted)| Recall (micro)| Recall (macro)| F1-Score (weighted) | F1-Score (micro) | F1-Score (macro) |
|:----------------------------------------------------------------------------------------------------:|:-----------------:|:------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:----------------:|:-------------:|:-------------:|:-------------------:|:----------------:|:----------------:|
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-peymaner-uncased.ipynb) | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-peymaner-uncased |  49.536  |       26.037        |      19.731      |       28.447     |      11.953      |     11.953    |     12.190    |       15.651        |      14.888      |      16.481      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-armanner-uncased.ipynb) | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-armanner-uncased |  46.824  |       20.188        |      17.988      |       21.610     |       9.448      |      9.448    |     11.227    |       12.318        |      12.388      |      13.933      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-ner-uncased.ipynb)      | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-ner-uncased      |  50.959  |       24.971        |      18.525      |       26.845     |      13.820      |     13.820    |     15.260    |       16.642        |      15.831      |      18.095      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-base-uncased-ner-peyma.ipynb)      | ParsBERT v2.0     | HooshvareLab/bert-fa-base-uncased-ner-peyma      |  45.553  |       34.009        |      33.280      |       36.473     |       8.737      |      8.737    |      8.973    |       13.879        |      13.840      |      14.372      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-base-uncased-ner-arman.ipynb)      | ParsBERT v2.0     | HooshvareLab/bert-fa-base-uncased-ner-arman      |  43.195  |       25.249        |      25.438      |       24.734     |       6.732      |      6.732    |      6.535    |       10.628        |      10.646      |      10.336      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_albert-fa-zwnj-base-v2-ner.ipynb)          | ALBERT v3.0       | HooshvareLab/albert-fa-zwnj-base-v2-ner          |  22.691  |       15.569        |      10.408      |       12.848     |       6.974      |      6.974    |      8.156    |        4.340        |       8.352      |       4.996      |
|[Link](notebooks/named-entity-recognition/NER_m3hrdadfi_albert-fa-base-v2-ner-peyma.ipynb)            | ALBERT-fa-base-v2 | m3hrdadfi/albert-fa-base-v2-ner-peyma            |  34.559  |       44.590        |      46.667      |       38.932     |       1.488      |      1.488    |      1.211    |        2.878        |       2.883      |       2.347      |
|[Link](notebooks/named-entity-recognition/NER_m3hrdadfi_albert-fa-base-v2-ner-arman.ipynb)            | ALBERT-fa-base-v2 | m3hrdadfi/albert-fa-base-v2-ner-arman            |  34.697  |       34.080        |      33.017      |       32.992     |       1.233      |      1.233    |      1.227    |        2.368        |       2.376      |       2.355      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-zwnj-base-ner.ipynb)               | ParsBERT v3.0     | HooshvareLab/bert-fa-zwnj-base-ner               |  51.810  |       17.916        |      15.502      |       17.965     |      17.655      |     17.655    |     20.397    |       17.253        |      16.508      |      18.165      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_distilbert-fa-zwnj-base-ner.ipynb)         | DistilBERT v3.0   | HooshvareLab/distilbert-fa-zwnj-base-ner         |  52.514  |       23.359        |      14.692      |       24.128     |      14.413      |     14.413    |     18.803    |       16.053        |      14.551      |      17.874      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_roberta-fa-zwnj-base-ner.ipynb)            | Roberta v3.0      | HooshvareLab/roberta-fa-zwnj-base-ner            |          |                     |                  |                  |                  |               |               |                     |                  |                  |

#### Arman+Peyma+WikiAnn
The following table shows the statistics of the entities within this dataset: 

| B-ORG | I-ORG | B-LOC | I-LOC | B-EVE | I-EVE | B-MON | I-MON | B-FAC | I-FAC | B-DAT | I-DAT | B-PRO | I-PRO | B-PCT | I-PCT | B-TIM | I-TIM | B-PER | I-PER |   O    |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:------:|
| 3216  | 3967  | 2886  |  858  |  256  |  888  |  98   |  263  |  248  |  408  |  407  |  568  |  318  |  296  |  94   |  141  |  43   |  78   | 2646  | 1707  | 178611 |

In the following table, we will report evaluation results for test set of combination of `Peyma`, `Arman`, and `WikiAnn` datasets:

|  Notebook                                                                                            |  Model Type       |             Model Name                           | Accuracy | Precision (weighted)| Precision (micro)| Precision (macro)| Recall (weighted)| Recall (micro)| Recall (macro)| F1-Score (weighted) | F1-Score (micro) | F1-Score (macro) |
|:----------------------------------------------------------------------------------------------------:|:-----------------:|:------------------------------------------------:|:--------:|--------------------:|:----------------:|:----------------:|:----------------:|:-------------:|:-------------:|:-------------------:|:----------------:|:----------------:|
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-peymaner-uncased.ipynb) | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-peymaner-uncased |  95.193  |       66.276        |      62.580      |       48.997     |      64.578      |     64.578    |     68.455    |       64.193        |      63.563      |      54.502      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-armanner-uncased.ipynb) | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-armanner-uncased |  97.092  |       75.941        |      74.793      |       66.474     |      66.122      |     66.122    |     66.163    |       70.362        |      70.191      |      65.467      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-base-parsbert-ner-uncased.ipynb)      | ParsBERT v1.0     | HooshvareLab/bert-base-parsbert-ner-uncased      |  96.355  |       70.313        |      71.580      |       52.633     |      65.292      |     65.292    |     43.253    |       67.238        |      68.292      |      46.386      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-base-uncased-ner-peyma.ipynb)      | ParsBERT v2.0     | HooshvareLab/bert-fa-base-uncased-ner-peyma      |  94.883  |       67.397        |      63.336      |       48.696     |      53.187      |     53.187    |     55.799    |       58.141        |      57.820      |      49.258      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-base-uncased-ner-arman.ipynb)      | ParsBERT v2.0     | HooshvareLab/bert-fa-base-uncased-ner-arman      |  96.658  |       74.226        |      72.279      |       64.485     |      63.144      |     63.144    |     66.803    |       67.565        |      67.403      |      64.109      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_albert-fa-zwnj-base-v2-ner.ipynb)          | ALBERT v3.0       | HooshvareLab/albert-fa-zwnj-base-v2-ner          |   8.108  |       18.428        |       1.387      |       13.002     |       1.589      |      1.589    |     0.7417    |        1.634        |       1.481      |       0.953      |
|[Link](notebooks/named-entity-recognition/NER_m3hrdadfi_albert-fa-base-v2-ner-peyma.ipynb)            | ALBERT-fa-base-v2 | m3hrdadfi/albert-fa-base-v2-ner-peyma            |  91.347  |       34.816        |      26.637      |       23.847     |       6.186      |      6.186    |     3.7435    |       10.181        |      10.040      |       6.254      |
|[Link](notebooks/named-entity-recognition/NER_m3hrdadfi_albert-fa-base-v2-ner-arman.ipynb)            | ALBERT-fa-base-v2 | m3hrdadfi/albert-fa-base-v2-ner-arman            |  93.672  |       57.996        |      55.165      |       43.243     |      26.513      |     26.513    |     19.825    |       35.742        |      35.814      |      26.471      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_bert-fa-zwnj-base-ner.ipynb)               | ParsBERT v3.0     | HooshvareLab/bert-fa-zwnj-base-ner               |  90.795  |       57.248        |      27.097      |       45.379     |      50.548      |     50.548    |     47.709    |       52.321        |      35.281      |      43.328      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_distilbert-fa-zwnj-base-ner.ipynb)         | DistilBERT v3.0   | HooshvareLab/distilbert-fa-zwnj-base-ner         |  95.779  |       60.292        |      59.931      |       55.099     |      49.280      |     49.280    |     41.171    |       53.855        |      54.086      |      45.728      |
|[Link](notebooks/named-entity-recognition/NER_HooshvareLab_roberta-fa-zwnj-base-ner.ipynb)            | Roberta v3.0      | HooshvareLab/roberta-fa-zwnj-base-ner            |          |                     |                  |                  |                  |               |               |                     |                  |                  |					

#### Evaluation based on each entity 
In [result file](evaluation-results/named-entity-recognition.xlsx), we reports the results of evaluating each model.
You can find the per entity evaluation results in this file. 

## Textual Thematic Similarity Task

### Sample Inference
```python
import torch
from textual_thematic_similarity import TextualThematicSimilarity

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = 'm3hrdadfi/bert-fa-base-uncased-wikinli'
tts_model = TextualThematicSimilarity(model_name=model_name, model_architecture="BertForSequenceClassification", label2id= {"dissimilar": 0, "similar": 1})

sentences_1 = [
    'در جریان انقلاب آلمان در سال های ۱۹۱۸ و ۱۹۱۹ او به برپایی تشکیلات فرایکورپس که سازمانی شبه نظامی برای سرکوب تحرکات انقلابی کمونیستی در اروپای مرکزی بود ، کمک کرد .	',
    'در جریان انقلاب آلمان در سال های ۱۹۱۸ و ۱۹۱۹ او به برپایی تشکیلات فرایکورپس که سازمانی شبه نظامی برای سرکوب تحرکات انقلابی کمونیستی در اروپای مرکزی بود ، کمک کرد .	',
    'شهر شیراز در بین سال های ۱۳۴۷ تا ۱۳۵۷ محل برگزاری جشن هنر شیراز بود .	', 
    'شهر شیراز در بین سال های ۱۳۴۷ تا ۱۳۵۷ محل برگزاری جشن هنر شیراز بود .	'
]
sentences_2 = [
    'کاناریس بعد از جنگ در ارتش باقی ماند ، اول به عنوان عضو فرایکورپس و سپس در نیروی دریایی رایش.در ۱۹۳۱ به درجه سروانی رسیده بود .	',
    'پسر سرهنگ وسل فرییتاگ لورینگوون به نام نیکی در مورد ارتباط کاناریس با بهم خوردن توطئه هیتلر برای اجرای آدمربایی و ترور پاپ پیوس دوازدهم در ایتالیا در ۱۹۷۲ در مونیخ شهادت داده است .	',
    'جشنواره ای از هنر نمایشی و موسیقی بود که از سال ۱۳۴۶ تا ۱۳۵۶ در پایان تابستان هر سال در شهر شیراز و تخت جمشید برگزار می شد .	',
    'ورزشگاه پارس با ظرفیت ۵۰ هزار تن که در جنوب شیراز واقع شده است .	'
]
tts_model.thematic_similarity_inference_seq_classification(sentences_1, sentences_2, device, max_length=tts_model.config.max_position_embeddings)
```

```python
import torch
from textual_thematic_similarity import TextualThematicSimilarity

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = 'm3hrdadfi/bert-fa-base-uncased-wikinli-mean-tokens'
tts_model = TextualThematicSimilarity(model_name=model_name, model_architecture="sentence-transformer")

sentences_1 = [
    'در جریان انقلاب آلمان در سال های ۱۹۱۸ و ۱۹۱۹ او به برپایی تشکیلات فرایکورپس که سازمانی شبه نظامی برای سرکوب تحرکات انقلابی کمونیستی در اروپای مرکزی بود ، کمک کرد .	',
    'در جریان انقلاب آلمان در سال های ۱۹۱۸ و ۱۹۱۹ او به برپایی تشکیلات فرایکورپس که سازمانی شبه نظامی برای سرکوب تحرکات انقلابی کمونیستی در اروپای مرکزی بود ، کمک کرد .	',
    'شهر شیراز در بین سال های ۱۳۴۷ تا ۱۳۵۷ محل برگزاری جشن هنر شیراز بود .	', 
    'شهر شیراز در بین سال های ۱۳۴۷ تا ۱۳۵۷ محل برگزاری جشن هنر شیراز بود .	'
]
sentences_2 = [
    'کاناریس بعد از جنگ در ارتش باقی ماند ، اول به عنوان عضو فرایکورپس و سپس در نیروی دریایی رایش.در ۱۹۳۱ به درجه سروانی رسیده بود .	',
    'پسر سرهنگ وسل فرییتاگ لورینگوون به نام نیکی در مورد ارتباط کاناریس با بهم خوردن توطئه هیتلر برای اجرای آدمربایی و ترور پاپ پیوس دوازدهم در ایتالیا در ۱۹۷۲ در مونیخ شهادت داده است .	',
    'جشنواره ای از هنر نمایشی و موسیقی بود که از سال ۱۳۴۶ تا ۱۳۵۶ در پایان تابستان هر سال در شهر شیراز و تخت جمشید برگزار می شد .	',
    'ورزشگاه پارس با ظرفیت ۵۰ هزار تن که در جنوب شیراز واقع شده است .	'
]
tts_model.thematic_similarity_inference_pair_similarity(sentences_1, sentences_2, device, label_list=["dissimilar", "similar"], similarity_threshold=0.5)
```

### Evaluation
We evaluated the available models on the following datasets: 

- [**Wiki D/Similar v1.0.0**](https://drive.google.com/uc?id=1P-KfNVIAx4HkaWFxc9aFoO3sHzHJFaVn): 
       Wiki D-Similar is another form of thematic similarity dataset with 137,402 records that tags pairs of sentences into a form of similar or dissimilar.
       We apply our evaluation on test set of this dataset which contains 5457 samples.
- [**Wiki Triplet v1.0.0**](https://drive.google.com/uc?id=1-lfrhHZwleYR4s0xGkXZPXxTeF25Q4C3): 
       A triplet-objective dataset extracted from Wikipedia Section Sentences into a triplet-form of anchor, positive and negative examples. 
       It covers 191,929 samples.
       We apply our evaluation on test set of this dataset which contains  samples.

All evaluation steps can be found in the [notebooks](notebooks/textual-thematic_similarity) associated with this task.
All the experimental results are aggregated in the corresponding [result file](evaluation-results/textual_thematic_similarity.xlsx).
This file contains information such as the hardware, the evaluation time, and the final results.

In the following table, we will report evaluation results for `Wiki D/Similar v1.0.0` dataset:

|  Notebook                                                                                                                      |  Model Type          |             Model Name                              | Accuracy | Precision (weighted)| Precision (macro)| Recall (weighted)| Recall (macro)| F1-Score (weighted) | F1-Score (macro) |
|:------------------------------------------------------------------------------------------------------------------------------:|:--------------------:|:---------------------------------------------------:|:--------:|:-------------------:|:----------------:|:----------------:|:-------------:|:-------------------:|:----------------:|
|[Link](notebooks/textual-thematic_similarity/TextualThematicSimilarity_m3hrdadfi_bert-fa-base-uncased-wikinli.ipynb)            | bert                 | m3hrdadfi/bert-fa-base-uncased-wikinli              |  76.642  |       76.722        |       76.721     |      76.642      |     76.642    |       76.625        |      76.625      |
|[Link](notebooks/textual-thematic_similarity/TextualThematicSimilarity_m3hrdadfi_bert-fa-base-uncased-wikinli-mean-tokens.ipynb)| Sentence-Transformer | m3hrdadfi/bert-fa-base-uncased-wikinli-mean-tokens  |  65.308  |       75.103        |       75.106     |      65.308      |     65.303    |       61.555        |      61.553      |

|  Notebook                                                                                                                      |  Model Type          |          Model Name                              |Cosine-Accuracy|Cosine-Accuracy_threshold|Cosine-Average Precision|Cosine-f1|Cosine-f1_threshold|Cosine-Precision|Cosine-Recall|Dot Product-Accuracy|Dot Product-Accuracy_threshold|Dot Product-Average Precision|Dot Product-f1|Dot Product-f1_threshold|Dot Product-Precision|Dot Product-Recall|Euclidean-Accuracy|Euclidean-Accuracy_threshold|Euclidean-Average Precision|Euclidean-f1|Euclidean-f1_threshold|Euclidean-Precision|Euclidean-Recall|Manhatten-Accuracy|Manhatten-Accuracy_threshold|Manhatten-Average Precision|Manhatten-f1|Manhatten-f1_threshold|Manhatten-Precision|Manhatten-Recall|
|:------------------------------------------------------------------------------------------------------------------------------:|:--------------------:|:------------------------------------------------:|:-------------:|:-----------------------:|:----------------------:|:-------:|:-----------------:|:--------------:|:-----------:|:------------------:|:----------------------------:|:---------------------------:|:------------:|:----------------------:|:-------------------:|:----------------:|:----------------:|:--------------------------:|:-------------------------:|:----------:|:--------------------:|:-----------------:|:--------------:|:----------------:|:--------------------------:|:-------------------------:|:----------:|:--------------------:|:-----------------:|:--------------:|
|[Link](notebooks/textual-thematic_similarity/TextualThematicSimilarity_m3hrdadfi_bert-fa-base-uncased-wikinli-mean-tokens.ipynb)| Sentence-Transformer |m3hrdadfi/bert-fa-base-uncased-wikinli-mean-tokens|     75.223    |        0.674464         |         81.149         |  77.007 |     0.632037      |      69.157    |    86.868   |       74.422       |          357.730804          |           81.147            |    76.217    |       324.5915222      |        67.241       |       87.959     |      75.259      |         18.482540          |           80.573          |   76.627   |      20.075726       |       67.991      |     87.777     |      75.168      |         404.822327         |          80.639           |   76.657   |       439.425171     |       67.951      |     87.923     |

In the following table, we will report evaluation results for `Wiki Triplet v1.0.0` dataset:

|  Notebook                                                                                                                          |  Model Type          |          Model Name                                  | Accuracy Cosine Distance | Accuracy Manhatten Distance|Accuracy Euclidean Distance|
|:----------------------------------------------------------------------------------------------------------------------------------:|:--------------------:|:----------------------------------------------------:|:------------------------:|:--------------------------:|:-------------------------:|
|[Link](notebooks/textual-thematic_similarity/TextualThematicSimilarity_m3hrdadfi_bert-fa-base-uncased-wikitriplet-mean-tokens.ipynb)| Sentence-Transformer |m3hrdadfi/bert-fa-base-uncased-wikitriplet-mean-tokens|          93.331          |          93.400            |          93.314           |
