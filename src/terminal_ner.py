from transformers import pipeline

ner = pipeline(
    "ner",
    model="../models/bert_ner",
    aggregation_strategy="simple"
)

print("NER System Ready")
print("Type 'exit' to quit\n")

while True:

    text = input("Enter text: ")

    if text.lower() == "exit":
        break

    results = ner(text)

    print("\nEntities Found:")

    for r in results:
        print(r["word"], "→", r["entity_group"])

    print("\n")