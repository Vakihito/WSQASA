## parameters ##

dataset_name = 'tweet_qa'
random_state = 42


dir_path = f'/content/drive/Shareddrives/question gen 2/pipe similarity fold extraction - gen issues - 7/{dataset_name}/'
data_dir_path = f'{dir_path}data/'


input_file_path = f'{data_dir_path}{dataset_name}_train_form.pkl'
output_file_path = f'{data_dir_path}{dataset_name}_syntatic_train.pkl'


# depencies
# !pip install gdown
# !pip install -U transformers==3.0.0
# !python -m nltk.downloader punkt
# !git clone https://github.com/patil-suraj/question_generation.git

# Commented out IPython magic to ensure Python compatibility.
# %cd question_generation

import os 
import shutil
from sklearn.model_selection import train_test_split
from pipelines import pipeline
from tqdm import tqdm
import pandas as pd
import re

df = pd.read_pickle(input_file_path)[['id', 'context']]
contexts = df['context'].unique()

df.head(1)

nlp = pipeline("question-generation")

ans_questions_map = {}
for context in tqdm(contexts):
  L_questions = []
  try:
    L_questions = nlp(context)
  except:
    pass
  ans_questions_map[context] = L_questions

L_id = []
L_context = []
L_question = []
L_possible_ans = []
for _, row in df.iterrows():
  for cur_pred in ans_questions_map[row['context']]:  
    L_id.append(row['id'])
    L_context.append(row['context'])
    L_question.append(cur_pred['question'])
    L_possible_ans.append(cur_pred['answer'])

df_question_gen = pd.DataFrame({'id' : L_id,
                                'context' : L_context,
                                'question' : L_question,
                                'answers': L_possible_ans})

df_question_gen.head(3)

import numpy as np

def find_string_in_text(cur_sub_str, cur_text):
  lower_cur_sub_str = cur_sub_str.lower()
  lower_cur_text = cur_text.lower()
  return lower_cur_text.find(lower_cur_sub_str)

def format_answer_col(cur_df, ans_col_name, text_col_name='text'):
  formated_ans_list = []

  all_ans   = cur_df[ans_col_name].values
  all_texts = cur_df[text_col_name].values

  for cur_ans_list, cur_text in zip(all_ans, all_texts):
    temp_formated_ans_list = { 'text':[], 'answer_start' : []}
    # print(type(cur_ans_list))
    if isinstance(cur_ans_list, list) or isinstance(cur_ans_list, np.ndarray):
      for cur_ans in cur_ans_list:
        start_pos = find_string_in_text(cur_ans, cur_text)
        if start_pos != -1:
          temp_formated_ans_list['text'].append(cur_ans)
          temp_formated_ans_list['answer_start'].append(start_pos)

    else:
      start_pos = find_string_in_text(cur_ans_list, cur_text)
      if start_pos != -1:
        temp_formated_ans_list['text'].append(cur_ans_list)
        temp_formated_ans_list['answer_start'].append(start_pos)
    
    formated_ans_list.append(temp_formated_ans_list)
  return formated_ans_list

df_question_gen['answers'] =  format_answer_col(df_question_gen, 'answers', 'context')
df_train_extractive = df_question_gen[df_question_gen['answers'].apply(lambda x: len(x['text']) > 0)]

df_train_extractive.head(5)

print(f"total size of the artificial dataset {len(df_train_extractive)} ")

df_train_extractive[['id', 'context', 'question', 'answers']].to_pickle(output_file_path)