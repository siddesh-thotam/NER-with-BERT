from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

model_path = "../models/bert_ner"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

text = "Elon Musk founded SpaceX in California."

results = ner(text)

for r in results:
    print(r["word"], "→", r["entity_group"], "score:", round(r["score"],3))