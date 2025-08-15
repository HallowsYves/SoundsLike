import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from ner.model.load_model_from_hub import get_ner_pipeline 
import os
"""
    Goes through the prompt and tokenizes it. Runs it through the trained NER model, and takes out
    inputs which can be labeled with BIO. It then puts them in their appropriate entity and flushes it.
"""
# Load model and tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root

# model_path = os.path.join(BASE_DIR, "models", "distilbert-ner")

ner_pipeline = get_ner_pipeline()



label_list = ["O", "B-MOOD", "B-SONG", "I-SONG", "B-ARTIST", "I-ARTIST"]

def extract_entities(text: str):
    """
    Convenience helper: returns a dict with grouped entities from the model,
    e.g., {"MOOD": ["chill"], "SONG": ["2031"], "ARTIST": ["Inner Wave"]}

    The HF pipeline with aggregation_strategy="simple" already merges tokens in
    the same entity, so each item is a whole span.
    """
    preds = ner_pipeline(text)  # list[dict]: entity_group, word, score, start, end
    out = {"MOOD": [], "SONG": [], "ARTIST": []}
    for p in preds:
        group = p.get("entity_group", "")
        if group in out:
            out[group].append(p["word"])
    # drop empty keys
    return {k: v for k, v in out.items() if v}