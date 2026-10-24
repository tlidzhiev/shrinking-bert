from collections.abc import Sequence

import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

LABEL_NAMES = [
    'O',
    'B-PER',
    'I-PER',
    'B-ORG',
    'I-ORG',
    'B-LOC',
    'I-LOC',
    'B-MISC',
    'I-MISC',
]
ID2LABEL = {i: label for i, label in enumerate(LABEL_NAMES)}
LABEL2ID = {label: i for i, label in enumerate(LABEL_NAMES)}

# Label id used by the losses and by `compute_metrics` to skip a position.
IGNORE_INDEX = -100


def load_conll2003() -> DatasetDict:
    dataset = load_dataset('eriktks/conll2003', trust_remote_code=True)
    return dataset.remove_columns(['id', 'pos_tags', 'chunk_tags'])  # ty:ignore[invalid-return-type]


def align_labels_with_tokens(tags: Sequence[int], word_ids: Sequence[int | None]) -> np.ndarray:
    word_index = np.array([-1 if w is None else w for w in word_ids], dtype=int)
    labels = np.full(len(word_index), IGNORE_INDEX, dtype=int)

    is_word = word_index >= 0
    positions = word_index[is_word]
    if positions.size == 0:
        return labels

    aligned = np.asarray(tags, dtype=int)[positions]
    is_continuation = np.concatenate([[False], positions[1:] == positions[:-1]])
    labels[is_word] = np.where(is_continuation & (aligned % 2 == 1), aligned + 1, aligned)
    return labels


def tokenize_and_align_labels(
    batch: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 180,
) -> dict:
    encoding = tokenizer(
        batch['tokens'],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    labels = [
        align_labels_with_tokens(tags, encoding.word_ids(i))
        for i, tags in enumerate(batch['ner_tags'])
    ]
    return {**encoding, 'labels': labels}


def get_tokenized_datasets(
    tokenizer: str = 'bert-base-cased',
    max_length: int = 180,
    batch_size: int = 256,
    max_samples: int | None = None,
) -> DatasetDict:
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    dataset = load_conll2003()
    if max_samples is not None:
        dataset = DatasetDict(
            {
                name: split.select(range(min(max_samples, len(split))))
                for name, split in dataset.items()
            }
        )
    return dataset.map(
        tokenize_and_align_labels,
        fn_kwargs={'tokenizer': _tokenizer, 'max_length': max_length},
        batched=True,
        batch_size=batch_size,
        remove_columns=['tokens', 'ner_tags'],
        desc='Tokenizing and aligning labels',
    )
