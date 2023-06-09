import os
## dataset parameters ##
dataset_name = os.environ['dataset_name']
domain = os.environ['domain']

## paths for files ##
dir_path = os.environ['dir_path']
data_dir_path = os.environ['data_dir_path']

## input and outputs ##
input_file_path_not_finetuned = os.environ['input_file_path_not_finetuned_5']
input_file_path_finetuned = os.environ['input_file_path_finetuned_5']

output_file_path = os.environ['output_file_path_5']

print(" \033[94m[INFO]\033[0m dataset_name : ", dataset_name) 
print(" \033[94m[INFO]\033[0m domain : ", domain) 
print(" \033[94m[INFO]\033[0m dir_path : ", dir_path) 
print(" \033[94m[INFO]\033[0m data_dir_path : ", data_dir_path) 
print(" \033[94m[INFO]\033[0m input_file_path_not_finetuned : ", input_file_path_not_finetuned) 
print(" \033[94m[INFO]\033[0m input_file_path_finetuned : ", input_file_path_finetuned) 
print(" \033[94m[INFO]\033[0m output_file_path : ", output_file_path) 

import numpy as np
import pandas as pd
from evaluate import load

squad_metric = load("squad")
meteor_metric = load("meteor")
bleu_metric = load("bleu")

def find_string_in_text(cur_sub_str, cur_text):
  lower_cur_sub_str = cur_sub_str.lower()
  lower_cur_text = cur_text.lower()
  return lower_cur_text.find(lower_cur_sub_str)

def format_answer_col(cur_df, ans_col_name, text_col_name='text'):
  """
    returns in the answers in format of the squad dataset:
     {
       text : [ list of answers ],
       answer_start : [list of intergers]
     }
  """
  
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

df_test_finetuned = pd.read_pickle(input_file_path_finetuned)
df_test_not_finetuned = pd.read_pickle(input_file_path_not_finetuned)

df_test_finetuned['answers_formated_for_squad_metrics'] = format_answer_col(df_test_finetuned, 'extractive_answers', 'context')
df_test_not_finetuned['answers_formated_for_squad_metrics'] = format_answer_col(df_test_not_finetuned, 'extractive_answers', 'context')

df_test_not_finetuned.head(1)

df_test_finetuned.head(1)

assert np.all(df_test_finetuned[df_test_finetuned.columns[:4]].values == df_test_not_finetuned[df_test_finetuned.columns[:4]].values)

def get_metrics_for_df(cur_df, model_name):
  answers_prediction = cur_df['answer_prediction'].values
  answers_reference = cur_df['extractive_answers'].values


  formatted_predictions_squad = [
    {"id": str(k), "prediction_text": v} for k, v in cur_df[['id', 'answer_prediction']].values
  ]
  references_squad = [
      {"id": str(k), "answers": v} for k, v in cur_df[['id', 'answers_formated_for_squad_metrics']].values
  ]

  squad_metrics_results = squad_metric.compute(predictions=formatted_predictions_squad, references=references_squad)

  blue_ngram_1_results = bleu_metric.compute(predictions=answers_prediction, references=answers_reference, max_order=1)
  blue_ngram_2_results = bleu_metric.compute(predictions=answers_prediction, references=answers_reference, max_order=2)
  blue_ngram_3_results = bleu_metric.compute(predictions=answers_prediction, references=answers_reference, max_order=3)
  blue_ngram_4_results = bleu_metric.compute(predictions=answers_prediction, references=answers_reference, max_order=4)

  meteor_metric_results = meteor_metric.compute(predictions=answers_prediction, references=answers_reference)

  dict_metrics = {
      'dataset' : dataset_name,
      'model_name' : model_name,
      'exact_match' :  [squad_metrics_results['exact_match'] / 100],
      'f1' :           [squad_metrics_results['f1'] / 100],
      'bleu_1_ngram' : [blue_ngram_1_results['bleu']],
      'bleu_2_ngram' : [blue_ngram_2_results['bleu']],
      'bleu_3_ngram' : [blue_ngram_3_results['bleu']],
      'bleu_4_ngram' : [blue_ngram_4_results['bleu']],
      'meteor' :       [meteor_metric_results['meteor']],
  }

  return pd.DataFrame( dict_metrics)

results_not_finetuned = get_metrics_for_df(df_test_not_finetuned, 'not finetuned')
results_not_finetuned

results_finetuned = get_metrics_for_df(df_test_finetuned, 'finetuned')
results_finetuned

df_results = pd.concat([results_not_finetuned, results_finetuned])
df_results['total_test_data'] = len(df_test_finetuned)

print("total data on test : ", len(df_test_finetuned))

df_results.to_csv(output_file_path, index=False)

print("\033[92mjob success\033[0m\n\n")