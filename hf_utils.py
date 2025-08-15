# hf_utils.py
import os, json
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_DATASET_REPO = "HallowsYves/soundslike-data"  
HF_DATASET_REV  = os.getenv("HF_DATASET_REV", "main") 
HF_TOKEN        = st.secrets.get("HF_TOKEN", None)      

@st.cache_data(show_spinner=False)
def _hub_path(filename: str,
              repo_id: str = HF_DATASET_REPO,
              repo_type: str = "dataset",
              revision: str = HF_DATASET_REV):
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        token=HF_TOKEN,     
    )

@st.cache_data(show_spinner=False)
def load_csv_from_hub(filename: str, **read_csv_kwargs) -> pd.DataFrame:
    return pd.read_csv(_hub_path(filename), **read_csv_kwargs)

@st.cache_data(show_spinner=False)
def load_json_from_hub(filename: str):
    with open(_hub_path(filename), "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_binary_from_hub(filename: str) -> bytes:
    with open(_hub_path(filename), "rb") as f:
        return f.read()
