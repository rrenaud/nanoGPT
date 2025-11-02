import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenizer_loader import build_tokenizer

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

parser = argparse.ArgumentParser(description="Prepare the Shakespeare dataset with a chosen tokenizer.")
parser.add_argument(
    "--tokenizer",
    choices=("gpt2", "superbpe"),
    default="gpt2",
    help="Tokenizer to use when encoding the dataset.",
)
parser.add_argument(
    "--tokenizer_repo",
    default=None,
    help="Optional HuggingFace repository id for tokenizer assets (SuperBPE only).",
)
args = parser.parse_args()

tokenizer = build_tokenizer(
    args.tokenizer,
    tokenizer_repo=args.tokenizer_repo,
)
print(f"Using tokenizer '{tokenizer.name}' (EOS id {tokenizer.eos_token_id})")

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_ids = tokenizer.encode_ordinary(train_data)
val_ids = tokenizer.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

token_dtype = (
    np.uint16
    if tokenizer.max_token_id <= np.iinfo(np.uint16).max
    else np.uint32
)

# export to bin files
train_ids = np.array(train_ids, dtype=token_dtype)
val_ids = np.array(val_ids, dtype=token_dtype)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

meta = {
    'tokenizer': tokenizer.name,
    'tokenizer_repo': args.tokenizer_repo,
    'eos_token_id': tokenizer.eos_token_id,
    'max_token_id': tokenizer.max_token_id,
    'dtype': np.dtype(token_dtype).name,
    'vocab_size': tokenizer.vocab_size,
}
meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
