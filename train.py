from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding, BertForSequenceClassification, TrainingArguments, Trainer
import evaluate
import pandas as pd

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = [item for sublist in predictions for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    return accuracy.compute(predictions=predictions, references=labels)

data = load_from_disk("votes")

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_data = data.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

id2label = pd.read_pickle("people.pkl")[1].to_dict()
label2id = res = dict((v,k) for k,v in id2label.items())

model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=len(id2label.keys()), id2label=id2label, label2id=label2id, problem_type="multi_label_classification"
)

training_args = TrainingArguments(
    output_dir="training_out",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    num_train_epochs=50,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("./fine_tuned_model")