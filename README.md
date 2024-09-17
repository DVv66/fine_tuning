# Sentiment Analysis Fine-Tuning with GPT-2 on Rotten Tomatoes Dataset
This repository contains code for fine-tuning a GPT-2 language model on the Rotten Tomatoes dataset for sentiment analysis. The goal is to train a model that can predict the sentiment (positive or negative) of movie reviews.

## Introduction
This project demonstrates how to fine-tune a causal language model (GPT-2) on a classification task (sentiment analysis) using the Hugging Face Transformers library. It involves customizing the data processing to fit the causal language modeling paradigm and setting up the training loop with appropriate hyperparameters.

## Dataset
We use the Rotten Tomatoes dataset from the Hugging Face Datasets library. This dataset contains movie reviews labeled as positive (1) or negative (0).

## Model
We fine-tune the pre-trained GPT-2 model (gpt2) from Hugging Face. GPT-2 is a causal language model, and we adapt it for sentiment analysis by formulating the task as next-token prediction.

## Data Preprocessing
To adapt the dataset for training GPT-2 on sentiment analysis, we perform the following steps:

Define Label Tokens

We map the sentiment labels to specific tokens:

```
named_labels = ['neg', 'pos']
label_ids = [
    tokenizer(named_labels[i], add_special_tokens=False)["input_ids"][0]
    for i in range(len(named_labels))
]
```

Process Function

We define a function process_fn to process the raw data:

```
def process_fn(examples):
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for i in range(len(examples[DATA_BODY_KEY])):
        prompt = f"{examples[DATA_BODY_KEY][i]} Sentiment: "
        inputs = tokenizer(prompt, add_special_tokens=False)
        label = label_ids[examples[DATA_LABEL_KEY][i]]
        input_ids = inputs["input_ids"] + [label]

        raw_len = len(input_ids)

        if raw_len >= MAX_LEN:
            input_ids = input_ids[-MAX_LEN:]
            attention_mask = [1] * MAX_LEN
            labels = [-100] * (MAX_LEN - 1) + [label]
        else:
            input_ids = input_ids + [tokenizer.pad_token_id] * (MAX_LEN - raw_len)
            attention_mask = [1] * raw_len + [0] * (MAX_LEN - raw_len)
            labels = [-100] * (raw_len - 1) + [label] + [-100] * (MAX_LEN - raw_len)
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
    return model_inputs
```

Prompt Construction: Combines the review text with "Sentiment:" to form the input prompt.

Tokenization: Converts text to token IDs without adding special tokens.

Label Appending: Adds the label token ID to the end of input_ids.

Padding/Truncation: Adjusts sequences to a fixed MAX_LEN.

Attention Mask: Indicates which tokens are actual data and which are padding.

Labels for Loss: Sets non-label tokens in labels to -100 to ignore them in loss computation.

Apply Processing to Datasets

```
tokenized_train_dataset = raw_train_dataset.map(
    process_fn,
    batched=True,
    remove_columns=columns,
    desc="Processing train dataset",
)

tokenized_valid_dataset = raw_valid_dataset.map(
    process_fn,
    batched=True,
    remove_columns=columns,
    desc="Processing validation dataset",
)
```

## Training
Data Collator

We use DataCollatorWithPadding to handle dynamic padding:

```
collater = DataCollatorWithPadding(
    tokenizer=tokenizer, return_tensors="pt",
)
```

Model Initialization

Load the pre-trained GPT-2 model and enable gradient checkpointing:
```
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.gradient_checkpointing_enable()
```

Training Arguments

Define training hyperparameters:
```
training_args = TrainingArguments(
    output_dir=f"./output-lr{args.lr}-batch{args.batch_size}",
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
```

Compute Metrics

Define a function to compute accuracy:
```
def compute_metric(eval_predictions):
    predictions, labels = eval_predictions

    label_indices = (labels != -100).nonzero()
    actual_labels = labels[label_indices]

    label_indices = (label_indices[0], label_indices[1] - 1)
    selected_logits = predictions[label_indices]

    predicted_labels = selected_logits[:, label_ids].argmax(axis=-1)
    predicted_labels = np.array(label_ids)[predicted_labels]

    correct_predictions = (predicted_labels == actual_labels).sum()
    accuracy = correct_predictions / len(actual_labels)

    return {"accuracy": accuracy}
```
Initialize Trainer
```
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collater,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    compute_metrics=compute_metric,
)
```
Start Training
```
trainer.train()
```
## Evaluation
The model is evaluated during training at specified intervals using the validation dataset. The accuracy metric is computed based on the predictions.
