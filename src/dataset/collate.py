from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from .dataset import IGNORE_INDEX

# Columns produced by the tokenizer that the model consumes directly.
TOKENIZER_COLUMNS = ('input_ids', 'token_type_ids', 'attention_mask')


class NerDataCollator:
    """Pad a batch of variable-length examples to the longest sequence in it."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, label_pad_token_id: int = IGNORE_INDEX):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[Any]) -> dict[str, Any]:
        batch = self.tokenizer.pad(
            [{key: row[key] for key in TOKENIZER_COLUMNS if key in row} for row in features],
            padding=True,
            return_tensors='pt',
        )
        width = batch['input_ids'].shape[1]

        if 'labels' in features[0]:
            batch['labels'] = torch.tensor(
                [
                    list(row['labels']) + [self.label_pad_token_id] * (width - len(row['labels']))
                    for row in features
                ],
                dtype=torch.long,
            )

        return dict(batch)
