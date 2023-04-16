import os
## dataset parameters ##

dataset_name = os.environ['dataset_name']
min_thold = float(os.environ['min_thold'] )
min_sentiment_thold = float(os.environ['min_sentiment_thold'])

## paths for files ##
dir_path = os.environ['dir_path']
data_dir_path = os.environ['data_dir_path']
model_dir_path = os.environ['model_dir_path']


input_file_path = os.environ['input_file_train_data'] 
input_file_issues_path = os.environ['input_file_issues_path'] 
output_file_path = os.environ['output_file_path_1']

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

if not os.path.exists(data_dir_path):
  os.mkdir(data_dir_path)

if not os.path.exists(model_dir_path):
  os.mkdir(model_dir_path)

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
for cur_ans in df['answers']:
  unique_answers += cur_ans['text'] 
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
for cur_ans_d in df['answers']:
  cur_dist, closest_issue = map_ans_to_issue[cur_ans_d['text'][0]]  ## there is only one answer per text
  L_distances.append(cur_dist)
  L_closest_issue.append(closest_issue)

df['issue_distance'] = L_distances
df['closest_issue'] = L_closest_issue

df = df[df['issue_distance'] > min_thold].reset_index(drop=True)

"""## Sentiment based filtering

---
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sid_obj = SentimentIntensityAnalyzer()

L_sentiment = []
for cur_ans_L in tqdm(df['answers'], total=len(df)):
  temp_sentiments = []
  for cur_ans_u in cur_ans_L['text']:
    cur_polarity = sid_obj.polarity_scores(cur_ans_u)['compound']
    temp_sentiments.append(cur_polarity)
  L_sentiment.append(np.array(temp_sentiments))

df['sentiment_polarity'] = L_sentiment

if min_sentiment_thold > 0:
  df = df[df['sentiment_polarity'].apply(lambda x: x[0]) > min_sentiment_thold].reset_index(drop=True)
else :
  df = df[df['sentiment_polarity'].apply(lambda x: x[0]) < min_sentiment_thold].reset_index(drop=True)

df[['id', 'context', 'question', 'answers', 'dataset']]

df[['id', 'context', 'question', 'answers', 'dataset']].to_pickle(output_file_path)

print("\033[92mjob success\033[0m\n\n")
