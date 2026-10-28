from pathlib import Path

import torch
from datasets import Dataset, Sequence, Value
from datasets.fingerprint import Hasher
from transformers import DataCollatorForTokenClassification, PreTrainedModel

from src.dataset import LABEL_NAMES

TEACHER_LOGITS = 'teacher_logits'
MODEL_INPUTS = ('input_ids', 'token_type_ids', 'attention_mask')


class DataCollatorWithTeacherLogits(DataCollatorForTokenClassification):
    def torch_call(self, features):
        has_logits = TEACHER_LOGITS in features[0]
        logits = [feature.pop(TEACHER_LOGITS) for feature in features] if has_logits else None
        batch = super().torch_call(features)
        if logits is not None:
            if self.tokenizer.padding_side != 'right':
                raise ValueError('teacher_logits padding supports right-padded batches only.')
            padded = torch.zeros(len(logits), batch['input_ids'].shape[1], len(LABEL_NAMES))
            for i, row in enumerate(logits):
                padded[i, : len(row)] = torch.tensor(row)
            batch[TEACHER_LOGITS] = padded
        return batch


@torch.no_grad()
def add_teacher_logits(
    dataset: Dataset,
    teacher: PreTrainedModel,
    teacher_path: str | Path,
    collator: DataCollatorForTokenClassification,
    batch_size: int = 128,
    device: str | torch.device = 'cpu',
) -> Dataset:
    weights = sorted(Path(teacher_path).glob('*.safetensors'))
    fingerprint = Hasher.hash(
        [dataset._fingerprint, [(w.name, w.stat().st_size, w.stat().st_mtime_ns) for w in weights]]
    )

    def teacher_logits(batch: dict) -> dict:
        features = [
            {key: batch[key][i] for key in MODEL_INPUTS if key in batch}
            for i in range(len(batch['input_ids']))
        ]
        inputs = {k: v.to(device) for k, v in collator(features).items()}
        logits = teacher(**inputs).logits.float().cpu().numpy()
        return {TEACHER_LOGITS: [logits[i, : len(f['input_ids'])] for i, f in enumerate(features)]}

    teacher.to(device).eval()  # ty:ignore[invalid-argument-type]
    features = dataset.features.copy()
    features[TEACHER_LOGITS] = Sequence(Sequence(Value('float32')))
    return dataset.map(
        teacher_logits,
        batched=True,
        batch_size=batch_size,
        features=features,
        new_fingerprint=fingerprint,
        desc='Computing teacher logits',
    )
