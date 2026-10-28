from transformers import AutoConfig, AutoModelForTokenClassification, PreTrainedModel

from src.dataset import ID2LABEL, LABEL2ID, LABEL_NAMES

from .factorization import factorize_embeddings, factorize_encoder_linears
from .sharing import share_encoder_layers, uniform_groups


def load_ner_model(name: str = 'bert-base-cased', **kwargs) -> PreTrainedModel:
    return AutoModelForTokenClassification.from_pretrained(
        name,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        **kwargs,
    )


def init_ner_model(name: str = 'bert-base-cased', **config_overrides) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(
        name,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        **config_overrides,
    )
    return AutoModelForTokenClassification.from_config(config)


def get_model(
    name: str = 'bert-base-cased',
    pretrained: bool = True,
    config_overrides: dict | None = None,
    embedding_rank: int | None = None,
    linear_rank: int | None = None,
    num_unique_layers: int | None = None,
) -> PreTrainedModel:
    if pretrained:
        model = load_ner_model(name)
    else:
        model = init_ner_model(name, **(config_overrides or {}))

    if embedding_rank is not None:
        model = factorize_embeddings(model, rank=embedding_rank)

    if linear_rank is not None:
        model = factorize_encoder_linears(model, rank=linear_rank)

    if num_unique_layers is not None:
        groups = uniform_groups(model.config.num_hidden_layers, num_unique_layers)
        model = share_encoder_layers(model, groups)
        print(f'Layer groups: {groups}')

    return model
