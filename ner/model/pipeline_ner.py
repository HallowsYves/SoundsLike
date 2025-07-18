import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

# Load model and tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Adjust to your project root

model_path = os.path.join(BASE_DIR, "models", "distilbert-ner")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Load model and tokenizer

label_list = ["O", "B-MOOD", "B-SONG", "I-SONG", "B-ARTIST", "I-ARTIST"]

def ner_pipeline(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)[0].tolist()  # batch size 1

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Map tokens to labels
    labels = [label_list[pred] for pred in predictions]

    # Group tokens by entities (simple BIO scheme)
    entities = {"mood": [], "song": [], "artist": []}
    current_entity = None
    current_tokens = []

    def flush_entity():
        nonlocal current_entity, current_tokens
        if current_entity and current_tokens:
            entity_text = tokenizer.convert_tokens_to_string(current_tokens)
            entities[current_entity].append(entity_text)
            current_entity = None
            current_tokens = []

    for token, label in zip(tokens, labels):
        if label == "O":
            flush_entity()
        elif label.startswith("B-"):
            flush_entity()
            current_entity = label[2:].lower()
            current_tokens = [token]
        elif label.startswith("I-") and current_entity == label[2:].lower():
            current_tokens.append(token)
        else:
            flush_entity()

    # Flush last entity if exists
    flush_entity()

    # Clean up lists to single strings or None if empty
    for k, v in entities.items():
        entities[k] = " ".join(v) if v else None

    return entities


# Example usage
prompt = "I want happy songs like Happy by Pharell Williams"
result = ner_pipeline(prompt)
print(result)
