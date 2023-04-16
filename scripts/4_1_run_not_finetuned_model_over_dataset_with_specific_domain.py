## dataset parameters ##
import os
dataset_name = os.environ['dataset_name']
domain = os.environ['domain']

## paths for files ##
dir_path = os.environ['dir_path']
data_dir_path = os.environ['data_dir_path']

## input and outputs ##
input_file_path = os.environ['input_file_path_4_1']

output_file_path = os.environ['output_file_path_4_1']

## tokenizer parameters ##
model_name = os.environ['model_checkpoint']
max_length = int(os.environ['max_length'])  # The maximum length of a feature (question and context)
doc_stride = int(os.environ['doc_stride'])  # The allowed overlap between two part of the context when splitting is performed.

## prediction parameters ##
batch_size = int(os.environ['batch_size'])


print(" \033[92m[INFO]\033[0m dataset_name : ", dataset_name)
print(" \033[92m[INFO]\033[0m domain : ", domain)
print(" \033[92m[INFO]\033[0m dir_path : ", dir_path)
print(" \033[92m[INFO]\033[0m data_dir_path : ", data_dir_path)
print(" \033[92m[INFO]\033[0m input_file_path : ", input_file_path)
print(" \033[92m[INFO]\033[0m output_file_path : ", output_file_path)
print(" \033[92m[INFO]\033[0m model_name : ", model_name)
print(" \033[92m[INFO]\033[0m max_length : ", max_length)
print(" \033[92m[INFO]\033[0m doc_stride : ", doc_stride)
print(" \033[92m[INFO]\033[0m batch_size : ", batch_size)

from datasets import load_dataset, Dataset
from IPython.display import display, HTML
from datasets import ClassLabel, Sequence
import pandas as pd
import random

df_test = pd.read_pickle(input_file_path)
df_test

datasets = Dataset.from_pandas(df_test)
datasets

from transformers import AutoModelForQuestionAnswering
from transformers import create_optimizer
from transformers import AutoTokenizer
import tensorflow as tf
from tqdm import tqdm

model = AutoModelForQuestionAnswering.from_pretrained(model_name) # loading the model
tokenizer = AutoTokenizer.from_pretrained(model_name)                # loading the tokenizer
pad_on_right = tokenizer.padding_side == "right"

from transformers import pipeline
model.cpu()
question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)

all_answers = []
questions = list(df_test['question'].values)
context = list(df_test['context'].values)
                
for i in tqdm(range(0, len(context), batch_size)):
  answer = question_answerer(question=questions[i: i + batch_size], context=context[i: i + batch_size])
  all_answers+=answer

answer_prediction = []
nswer_prediction_score = []
for i in range(len(context)):
  if 'answer' in all_answers[i]:
    answer_prediction.append(all_answers[i]['answer'])
    nswer_prediction_score.append(all_answers[i]['score'])
  else:
    answer_prediction.append(all_answers[i])
    nswer_prediction_score.append(0)

df_test['answer_prediction'] = answer_prediction
df_test['nswer_prediction_score'] = nswer_prediction_score

df_test

df_test.to_pickle(output_file_path)

print("\033[92mjob success\033[0m\n\n")