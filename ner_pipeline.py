import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from huggingface_hub import snapshot_download

DEFAULT_REPO = "HallowsYves/soundslike-ner"

def load_model_from_hf(repo_id: str = DEFAULT_REPO) -> str:
    """Download the NER model from Hugging Face and return the local path."""
    return snapshot_download(repo_id=repo_id)

# Load model and tokenizer
model_path = load_model_from_hf()
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

label_list = ["O", "B-MOOD", "B-SONG", "I-SONG", "B-ARTIST", "I-ARTIST"]

def ner_pipeline(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [label_list[pred] for pred in predictions]
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

    flush_entity()

    for k, v in entities.items():
        entities[k] = " ".join(v) if v else None

    return entities
