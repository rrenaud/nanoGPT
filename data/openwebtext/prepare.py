"""
Prepare the OpenWebText dataset and serialize it with a configurable tokenizer.
"""

import argparse
import os
import pickle

import numpy as np
from datasets import load_dataset  # huggingface datasets
from tqdm import tqdm

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenizer_loader import build_tokenizer

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
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

    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = tokenizer.encode_ordinary(example['text'])  # ignores any special tokens
        ids.append(tokenizer.eos_token_id)  # add end of text token
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    token_dtype = (
        np.uint16
        if tokenizer.max_token_id <= np.iinfo(np.uint16).max
        else np.uint32
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        arr = np.memmap(filename, dtype=token_dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    dtype_name = np.dtype(token_dtype).name
    meta = {
        'tokenizer': tokenizer.name,
        'tokenizer_repo': args.tokenizer_repo,
        'eos_token_id': tokenizer.eos_token_id,
        'max_token_id': tokenizer.max_token_id,
        'dtype': dtype_name,
        'vocab_size': tokenizer.vocab_size,
    }
    meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
