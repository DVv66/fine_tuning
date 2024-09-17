import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers import Trainer, Seq2SeqTrainer
import transformers
from transformers import DataCollatorWithPadding
from transformers import TextGenerationPipeline
import torch
import numpy as np
import os, re
from tqdm import tqdm
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, help="learning rate", default=2e-5)
parser.add_argument("--batch_size", type=int, help="batch size", default=8)
parser.add_argument("--warmup", type=float, help="warmup ratio", default=0.1)
parser.add_argument("--seed", type=int, help="random seed", default=42)
parser.add_argument("--interval", type=int, help="step interval for log, eval & save", default=100)
args = parser.parse_args()

MAX_LEN=32
MODEL_NAME = "gpt2"
# MODEL_NAME = "gpt2-large"

DATASET_NAME = "rotten_tomatoes"
DATA_BODY_KEY = "text"
DATA_LABEL_KEY = "label"

# Load data set
raw_datasets = load_dataset(DATASET_NAME)

# Training set
raw_train_dataset = raw_datasets["train"]

# Validation set
raw_valid_dataset = raw_datasets["validation"]


columns = raw_train_dataset.column_names

# Set random seeds
transformers.set_seed(args.seed)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = 0

named_labels = ['neg','pos']

label_ids = [
    tokenizer(named_labels[i],add_special_tokens=False)["input_ids"][0] 
    for i in range(len(named_labels))
]

# Define data processing functions that convert raw data to input_ids, attention_mask, labels
def process_fn(examples):
    model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
    for i in range(len(examples[DATA_BODY_KEY])):
        prompt = f"{examples[DATA_BODY_KEY][i]} Sentiment: "
        inputs = tokenizer(prompt,add_special_tokens=False)
        label = label_ids[examples[DATA_LABEL_KEY][i]]
        input_ids = inputs["input_ids"] + [label]
        
        raw_len = len(input_ids)

        if raw_len >= MAX_LEN:
            input_ids = input_ids[-MAX_LEN:]
            attention_mask = [1] * MAX_LEN
            labels = [-100]*(MAX_LEN - 1) + [label]
        else:
            input_ids = input_ids + [tokenizer.pad_token_id] * (MAX_LEN-raw_len)
            attention_mask = [1] * raw_len + [0] * (MAX_LEN-raw_len)
            labels = [-100] * (raw_len-1) + [label] + [-100] * (MAX_LEN-raw_len)
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
    return model_inputs


# Processing training data sets
tokenized_train_dataset = raw_train_dataset.map(
    process_fn,
    batched=True,
    remove_columns=columns,
    desc="Running tokenizer on train dataset",
)

# Process validation data sets
tokenized_valid_dataset = raw_valid_dataset.map(
    process_fn,
    batched=True,
    remove_columns=columns,
    desc="Running tokenizer on validation dataset",
)


# Define a data calibrator (automatically generate batch)
collater = DataCollatorWithPadding(
    tokenizer=tokenizer, return_tensors="pt",
)

# Definition model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,trust_remote_code=True)
# Save video memory
model.gradient_checkpointing_enable()

# Define training parameters
training_args = TrainingArguments(
    output_dir=f"./output-lr{args.lr}-batch{args.batch_size}",    # checkpoint save path
    evaluation_strategy="steps",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=args.batch_size,
    eval_steps=args.interval,
    logging_steps=args.interval,
    save_steps=args.interval,
    learning_rate=args.lr,
    warmup_ratio=args.warmup,
)
 

def compute_metric(eval_predictions):
    predictions, labels = eval_predictions

    label_indices = (labels != -100).nonzero()
    actual_labels = labels[label_indices]

    label_indices = (label_indices[0], label_indices[1]-1)
    selected_logits = predictions[label_indices]

    predicted_labels = selected_logits[:,label_ids].argmax(axis=-1)

    predicted_labels = np.array(label_ids)[predicted_labels]

    correct_predictions = (predicted_labels == actual_labels).sum()

    accuracy = correct_predictions / len(labels)

    return { "acc" : accuracy }


# Definition trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collater,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    compute_metrics=compute_metric,
)


# Start training
trainer.train()
