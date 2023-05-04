from transformers import BertForSequenceClassification, AutoTokenizer
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("Adopt Revised Code Volume 13a Of The South Carolina Code Of Laws, To The Extent Of Its Contents, As The Only General Permanent Statutory Law Of The State As Of January 1, 2023. - Ratified Title", return_tensors="pt")

model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) >= 0.5]
predicted_class_ids = predicted_class_ids.tolist()

for id in predicted_class_ids:
        print(model.config.id2label.get(id), end=", ")