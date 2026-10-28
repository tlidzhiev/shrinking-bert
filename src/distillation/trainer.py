from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer

from src.dataset import IGNORE_INDEX

from .precompute import TEACHER_LOGITS


class DistillationTrainer(Trainer):
    def __init__(self, *args, temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        teacher_logits = inputs.pop(TEACHER_LOGITS, None)
        outputs = model(**inputs)
        loss = outputs.loss

        if teacher_logits is not None:
            mask = inputs['labels'] != IGNORE_INDEX
            soft_loss = self.kl_loss(
                F.log_softmax(outputs.logits[mask].float() / self.temperature, dim=-1),
                F.softmax(teacher_logits[mask].float() / self.temperature, dim=-1),
            )
            loss = loss + self.temperature**2 * soft_loss

        return (loss, outputs) if return_outputs else loss
