from transformers import PreTrainedModel


def uniform_groups(num_layers: int, num_unique: int) -> list[list[int]]:
    if not 0 < num_unique <= num_layers:
        raise ValueError(f'num_unique must be in (0, {num_layers}], got {num_unique}')
    base, extra = divmod(num_layers, num_unique)
    groups, start = [], 0
    for i in range(num_unique):
        size = base + (i < extra)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


def share_encoder_layers(model: PreTrainedModel, groups: list[list[int]]) -> PreTrainedModel:
    encoder = model.base_model.encoder
    prefix = f'{model.base_model_prefix}.encoder.layer'
    layers = encoder.layer  # ty:ignore[unresolved-attribute]
    tied_weights_keys = dict(getattr(model, '_tied_weights_keys', None) or {})

    seen: set[int] = set()
    for group in groups:
        owner, *aliases = group
        if seen & set(group):
            raise ValueError(f'layer index appears in more than one group: {group}')
        seen.update(group)
        for index in aliases:
            layers[index] = layers[owner]  # ty:ignore[not-subscriptable, invalid-assignment]
            for name, _ in layers[owner].named_parameters():  # ty:ignore[not-subscriptable, unresolved-attribute]
                tied_weights_keys[f'{prefix}.{index}.{name}'] = f'{prefix}.{owner}.{name}'

    model._tied_weights_keys = tied_weights_keys
    return model
