from huggingface_hub import hf_hub_download
import pandas as pd
import json
from io import BytesIO

REPO_ID = "HallowsYves/soundslike-data"
REPO_TYPE = "dataset"

def _download(file_name: str) -> str:
    """Download a file from the Hugging Face dataset repo and return its local path."""
    return hf_hub_download(repo_id=REPO_ID, filename=file_name, repo_type=REPO_TYPE)

def load_csv_from_hf(file_name: str, index_col=None):
    """Load a CSV file from the Hugging Face dataset."""
    path = _download(file_name)
    return pd.read_csv(path, index_col=index_col)

def load_json_from_hf(file_name: str):
    """Load a JSON file from the Hugging Face dataset."""
    path = _download(file_name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_binary_from_hf(file_name: str):
    """Load a binary file from the Hugging Face dataset and return a BytesIO buffer."""
    path = _download(file_name)
    with open(path, "rb") as f:
        return BytesIO(f.read())
