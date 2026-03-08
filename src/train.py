from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score, precision_score, recall_score
from data.dataset_loader import load_conll_dataset
from src.preprocess import tokenize_and_align_labels
from src.config import *
dataset = load_conll_dataset()

# reduce dataset size
dataset["train"] = dataset["train"].select(range(2000))
dataset["validation"] = dataset["validation"].select(range(500))

label_list = dataset["train"].features["ner_tags"].feature.names
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="../models/results",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

data_collator = DataCollatorForTokenClassification(tokenizer)


import numpy as np

def compute_metrics(p):

    predictions, labels = p

    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred, label in zip(predictions, labels):

        cur_preds = []
        cur_labels = []

        for p_val, l_val in zip(pred, label):

            if l_val != -100:
                cur_preds.append(label_list[p_val])
                cur_labels.append(label_list[l_val])

        true_predictions.append(cur_preds)
        true_labels.append(cur_labels)

    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(MODEL_SAVE_PATH)