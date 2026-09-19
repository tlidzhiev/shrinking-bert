import logging

from transformers import AutoModelForTokenClassification, PreTrainedModel

from src.dataset import ID2LABEL, LABEL2ID, LABEL_NAMES

from .factorization import factorize_embeddings
from .sharing import share_encoder_layers, uniform_groups

logger = logging.getLogger(__name__)


def load_ner_model(name: str = 'bert-base-cased', **kwargs) -> PreTrainedModel:
    return AutoModelForTokenClassification.from_pretrained(
        name,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        **kwargs,
    )


def build_model(
    name: str = 'bert-base-cased',
    embedding_rank: int | None = None,
    num_unique_layers: int | None = None,
) -> PreTrainedModel:
    model = load_ner_model(name)

    if embedding_rank is not None:
        model = factorize_embeddings(model, rank=embedding_rank)

    if num_unique_layers is not None:
        groups = uniform_groups(model.config.num_hidden_layers, num_unique_layers)
        model = share_encoder_layers(model, groups)
        logger.info(f'Layer groups: {groups}')

    return model
