from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import torch
import json
import os

"""
    You have to run this twice, the first time to get model_output for all the logs
    The second time to get models/distibert... which contains the main info
    After you get the second one, you can delete model_output, or keep it
"""
# load
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def convert_to_dataset(data):
    return Dataset.from_list(data)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

train_data = load_json(os.path.join(BASE_DIR, "ner_data", "train.json"))
val_data = load_json(os.path.join(BASE_DIR, "ner_data", "val.json"))

# labels
label_list = ["O", "B-MOOD", "B-SONG", "I-SONG", "B-ARTIST", "I-ARTIST"]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

# tokenize
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align(example):
    tokenized = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    word_ids = tokenized.word_ids()
    labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != current_word:
            current_word = word_id
            labels.append(label_to_id[example["labels"][word_id]])
        else:
            labels.append(label_to_id[example["labels"][word_id]])
    tokenized["labels"] = labels
    return tokenized

dataset = Dataset.from_list(train_data)

tokenized_dataset = dataset.map(tokenize_and_align)

# model
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    num_labels=len(label_list), 
    id2label=id_to_label, 
    label2id=label_to_id
)

args = TrainingArguments(
    output_dir="./ner/model_output",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=10,
    logging_dir="./logs",
)

data_collator = DataCollatorForTokenClassification(tokenizer)

# load datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# tokenize
tokenized_train = train_dataset.map(tokenize_and_align, batched=False)
tokenized_val = val_dataset.map(tokenize_and_align, batched=False)

# combine
tokenized_dataset = DatasetDict({
    "train": tokenized_train,
    "validation": tokenized_val
})

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# train
trainer.train()


# evaluate
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[label] for label in label_seq if label != -100]
        for label_seq in labels
    ]
    true_predictions = [
        [label_list[pred] for pred, label in zip(pred_seq, label_seq) if label != -100]
        for pred_seq, label_seq in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Some redundancy here with "_" since we could use the for metrics
# Could look into it later
predictions, labels, _ = trainer.predict(tokenized_dataset["validation"])
metrics = compute_metrics((predictions, labels))
print(metrics)

# save
save_path = os.path.join(BASE_DIR, "models", "distilbert-ner")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)