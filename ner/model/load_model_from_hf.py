from huggingface_hub import snapshot_download

DEFAULT_REPO = "HallowsYves/soundslike-ner"

def load_model_from_hf(repo_id: str = DEFAULT_REPO) -> str:
    """Download the NER model from Hugging Face and return the local path."""
    model_path = snapshot_download(repo_id=repo_id)
    return model_path
