import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
from datasets import Dataset
from transformers import AutoTokenizer
from utils.label_utils import label2id

with open("data/ner/ner_train.json") as f:
    examples = json.load(f)

