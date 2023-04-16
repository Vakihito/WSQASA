import os
## dataset parameters ##

dataset_name = os.environ['dataset_name']
domain = os.environ['domain']
random_seed = int(os.environ['random_seed'])

## paths for files ##
dir_path = os.environ['dir_path']
data_dir_path = os.environ['data_dir_path']
model_dir_path = os.environ['model_dir_path']

input_file_path = os.environ['input_file_path_3']
output_model_path = os.environ['output_model_path_3']

## sentiment models parameters ##
model_checkpoint = os.environ['model_checkpoint']
batch_size = int(os.environ['batch_size'])

# tokenizer
max_length = int(os.environ['max_length'])  # The maximum length of a feature (question and context)
doc_stride = int(os.environ['doc_stride'])  # The allowed overlap between two part of the context when splitting is performed.

# hyper-parameters 
learning_rate = float(os.environ['learning_rate'])
num_train_epochs = int(os.environ['num_train_epochs'])
weight_decay = float(os.environ['weight_decay'])
encoder_layers_to_freeze = int(os.environ['encoder_layers_to_freeze'])


print(f' \033[92m[INFO]\033[0m dataset_name : ', dataset_name)
print(f' \033[92m[INFO]\033[0m domain : ', domain)
print(f' \033[92m[INFO]\033[0m random_seed : ', random_seed)
print(f" \033[92m[INFO]\033[0m dir_path : ",  dir_path)
print(f" \033[92m[INFO]\033[0m data_dir_path : ",  data_dir_path)
print(f" \033[92m[INFO]\033[0m model_dir_path : ",  model_dir_path)
print(f" \033[92m[INFO]\033[0m input_file_path : ",  input_file_path)
print(f" \033[92m[INFO]\033[0m output_model_path : ",  output_model_path)
print(f" \033[92m[INFO]\033[0m model_checkpoint : ",  model_checkpoint)
print(f" \033[92m[INFO]\033[0m batch_size : ",  batch_size)
print(f" \033[92m[INFO]\033[0m max_length : ",  max_length)
print(f" \033[92m[INFO]\033[0m doc_stride : ",  doc_stride)
print(f" \033[92m[INFO]\033[0m learning_rate : ",  learning_rate)
print(f" \033[92m[INFO]\033[0m num_train_epochs : ",  num_train_epochs)
print(f" \033[92m[INFO]\033[0m weight_decay : ",  weight_decay)

from datasets import load_dataset, Dataset
import tensorflow as tf
import torch
import pandas as pd
import numpy as np
import random

random.seed(random_seed)
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

df_syntatic_train = pd.read_pickle(input_file_path)
df_syntatic_train = df_syntatic_train[df_syntatic_train['dataset'] != dataset_name].reset_index(drop=True) ## filtering by dataset name

df_syntatic_train

assert not (dataset_name in list(df_syntatic_train['dataset'].unique()))

datasets = Dataset.from_pandas(df_syntatic_train)

datasets

"""### Preprocessing the data

---
"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

pad_on_right = tokenizer.padding_side == "right"

def prepare_train_features(examples):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

tokenized_datasets = datasets.map(
    prepare_train_features, batched=True, remove_columns=datasets.column_names
)

"""## Finetune the model

---
"""

from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()

from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

print(f"[INFO] freezing: {encoder_layers_to_freeze}")
if encoder_layers_to_freeze:
  if encoder_layers_to_freeze > 0:
    for params in model.distilbert.transformer.layer[encoder_layers_to_freeze:].parameters():
      params.requires_grad = False
  if encoder_layers_to_freeze < 0:
    for params in model.distilbert.transformer.layer[:-encoder_layers_to_freeze].parameters():
      params.requires_grad = False

training_args = TrainingArguments(
    output_dir="temp_model_output",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    disable_tqdm=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

import torch
import os
if not os.path.exists(output_model_path):
   os.mkdir(output_model_path)

torch.save(model.state_dict(), f'{output_model_path}/model_{dataset_name}.pt')
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)

print("[INFO] training Step Completed !")