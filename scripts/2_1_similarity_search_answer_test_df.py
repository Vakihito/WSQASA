import os

## dataset parameters ##
dataset_name = os.environ['dataset_name']
min_thold = float(os.environ['min_thold'] )
min_sentiment_thold = float(os.environ['min_sentiment_thold'])

## paths for files ##
dir_path = os.environ['dir_path']
data_dir_path = os.environ['data_dir_path']
model_dir_path = os.environ['model_dir_path']


input_file_path = os.environ['input_file_path_2']
input_file_issues_path = os.environ['input_file_issues_path']
output_file_path = os.environ['output_file_path_2']

model_name = os.environ['search_model_name']

print('\033[94m[INFO]\033[0m dataset_name : ', dataset_name)
print('\033[94m[INFO]\033[0m min_thold : ', min_thold)
print('\033[94m[INFO]\033[0m min_sentiment_thold : ', min_sentiment_thold)
print('\033[94m[INFO]\033[0m dir_path : ', dir_path)
print('\033[94m[INFO]\033[0m data_dir_path : ', data_dir_path)
print('\033[94m[INFO]\033[0m model_dir_path : ', model_dir_path)
print('\033[94m[INFO]\033[0m input_file_path : ', input_file_path)
print('\033[94m[INFO]\033[0m input_file_issues_path : ', input_file_issues_path)
print('\033[94m[INFO]\033[0m output_file_path : ', output_file_path)
print('\033[94m[INFO]\033[0m model_name : ', model_name)


import pandas as pd

df = pd.read_pickle(input_file_path)
df.dropna(inplace=True)
df.head()

len(df)

from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from nltk import ngrams
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import faiss
import os

sns.set_theme()

"""### Encoding Data"""

issues = pd.read_csv(input_file_issues_path)
issues = issues['issues'].unique()

unique_answers = []
for cur_ans in df['extractive_answers']:
  unique_answers += cur_ans
unique_answers = list(set(unique_answers))

encoder = SentenceTransformer(model_name, device='cuda')

encoded_issues = encoder.encode(issues, batch_size=128,show_progress_bar=True)
encoded_answers = encoder.encode(unique_answers, batch_size=128,show_progress_bar=True)

"""### Indexing data using FAISS

---
"""

res = faiss.StandardGpuResources()  # use a single GPU
dimension = encoded_answers[0].shape[0]

my_index = faiss.IndexFlatIP(dimension)
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, my_index)
gpu_index_flat.add(encoded_issues)

distances, indexes = gpu_index_flat.search(encoded_answers, k=1)

map_ans_to_issue = {cur_ans : (cur_dist, issues[cur_idx]) for cur_ans, cur_dist, cur_idx in zip(unique_answers, distances, indexes)}

L_distances, L_closest_issue = [], []
for cur_ans_L in df['extractive_answers']:
  temp_distances, temp_closest_issue = [], []
  for cur_ans_u in cur_ans_L:
    cur_dist, closest_issue = map_ans_to_issue[cur_ans_u]  
    temp_distances.append(cur_dist[0])
    temp_closest_issue.append(closest_issue[0])
  
  L_distances.append(temp_distances)
  L_closest_issue.append(temp_closest_issue)

df['issue_distance'] = L_distances
df['closest_issue'] = L_closest_issue

def keep_only_one_bigget_than_thold(cur_df, thold):
  """
    this function gets only answers with a specific polarity
  """  
  L_extractive_ans, L_ans_issue, L_ans_dist, L_id, L_context, L_question = ([] for i in range(len(cur_df.columns)))

  for _, cur_row in cur_df.iterrows(): 
    temp_extractive_ans, temp_ans_issue, temp_ans_dist = ([], [], [])
    for i in range(len(cur_row['issue_distance'])):
      if cur_row['issue_distance'][i] > thold:
        temp_extractive_ans.append(cur_row['extractive_answers'][i])
        temp_ans_issue.append(cur_row['closest_issue'][i])
        temp_ans_dist.append(cur_row['issue_distance'][i])
    if len(temp_extractive_ans) > 0:
      L_extractive_ans.append(np.array(temp_extractive_ans))
      L_ans_issue.append(np.array(temp_ans_issue))
      L_ans_dist.append(np.array(temp_ans_dist))
      L_id.append(cur_row['id'])
      L_context.append(cur_row['context'])
      L_question.append(cur_row['question'])
  return pd.DataFrame({'id': L_id, 'context': L_context, 'question': L_question, 
                       'extractive_answers': L_extractive_ans, 'closest_issue': L_ans_issue,
                       'issue_distance' : L_ans_dist})

df.columns

df_only_bigger = keep_only_one_bigget_than_thold(df, min_thold)

df_only_bigger.head(3)

"""## Sentiment based filtering

---
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sid_obj = SentimentIntensityAnalyzer()

L_sentiment = []
for cur_ans_L in tqdm(df_only_bigger['extractive_answers'], total=len(df_only_bigger)):
  temp_sentiments = []
  for cur_ans_u in cur_ans_L:
    cur_polarity = sid_obj.polarity_scores(cur_ans_u)['compound']
    temp_sentiments.append(cur_polarity)
  L_sentiment.append(np.array(temp_sentiments))

df_only_bigger['sentiment_polarity'] = L_sentiment
df_only_bigger.head(3)

def filter_only_negative_by_thold(cur_df, cur_thold):
  cur_df_temp = cur_df.copy()
  L_extractive_answers,	L_closest_issue,	L_issue_distance, L_sentiment_polarity = [], [], [], []
  for _, row in cur_df_temp.iterrows():
    if cur_thold > 0:
      idx_list_filter =  row['sentiment_polarity'] > min_sentiment_thold
    else:
      idx_list_filter =  row['sentiment_polarity'] < min_sentiment_thold
    L_extractive_answers.append( list(row['extractive_answers'][idx_list_filter]) )
    L_closest_issue.append( list(row['closest_issue'][idx_list_filter]))
    L_issue_distance.append( list(row['issue_distance'][idx_list_filter]))
    L_sentiment_polarity.append( list(row['sentiment_polarity'][idx_list_filter]))

  cur_df_temp['extractive_answers'] = L_extractive_answers
  cur_df_temp['closest_issue'] = L_closest_issue
  cur_df_temp['issue_distance'] = L_issue_distance
  cur_df_temp['sentiment_polarity'] = L_sentiment_polarity

  return cur_df_temp[cur_df_temp['sentiment_polarity'].apply(lambda x: len(x) > 0)].copy()

df_only_bigger = filter_only_negative_by_thold(df_only_bigger, min_sentiment_thold)

df_only_bigger.reset_index(inplace=True,drop=True)
df_only_bigger

df_only_bigger[['id', 'context', 'question', 'extractive_answers']]

df_only_bigger[['id', 'context', 'question', 'extractive_answers']].to_pickle(output_file_path)

print("\033[92mjob success\033[0m\n\n")