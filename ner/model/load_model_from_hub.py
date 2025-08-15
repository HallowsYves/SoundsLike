import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

REPO_ID = "HallowsYves/soundslike-ner"

@st.cache_resource(show_spinner=False)
def get_ner_pipeline():
    """
    Download & cache the tokenizer/model from Hugging Face Hub, then return
    a ready-to-use HF pipeline that aggregates B-/I- tags into whole entities.
    """
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForTokenClassification.from_pretrained(REPO_ID)

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  
        device=device,
    )

@st.cache_resource(show_spinner=False)
def get_label_list():
    """
    Returns the label list as defined in the model config, e.g.:
    ["O","B-MOOD","B-SONG","I-SONG","B-ARTIST","I-ARTIST"]
    """
    # Pull from config so you don’t need to hardcode it
    model = AutoModelForTokenClassification.from_pretrained(REPO_ID)
    id2label = model.config.id2label
    # sort by id to ensure stable order
    return [id2label[i] for i in sorted(id2label.keys(), key=int)]
