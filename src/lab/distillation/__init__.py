from .precompute import TEACHER_LOGITS, DataCollatorWithTeacherLogits, add_teacher_logits
from .trainer import DistillationTrainer

__all__ = [
    'TEACHER_LOGITS',
    'DataCollatorWithTeacherLogits',
    'DistillationTrainer',
    'add_teacher_logits',
]
