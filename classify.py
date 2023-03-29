from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Nah, it wasn't that great. I mean it was OK, but I didn't love it a lot.", return_tensors="pt")

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")

with torch.no_grad():
    logits = model(**inputs).logits


predicted_class_id = logits.argmax().item()
print(model.config.id2label[predicted_class_id])