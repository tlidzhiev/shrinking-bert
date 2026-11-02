import torch
from torch import nn
from transformers import PreTrainedModel


def low_rank_factors(matrix: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < rank <= min(matrix.shape):
        raise ValueError(f'rank must be in (0, {min(matrix.shape)}], got {rank}.')
    u, s, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    sqrt_s = torch.sqrt(s[:rank])
    a = u[:, :rank] * sqrt_s
    b = vh[:rank] * sqrt_s.unsqueeze(1)
    return a.to(matrix.dtype), b.to(matrix.dtype)


class FactorizedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, rank: int = 64):
        super().__init__()
        vocab_size, hidden_size = embedding.weight.shape
        self.num_embeddings = vocab_size
        self.embedding_dim = hidden_size
        self.rank = rank
        self.down = nn.Embedding(vocab_size, rank)
        self.up = nn.Linear(rank, hidden_size, bias=False)

        with torch.no_grad():
            a, b = low_rank_factors(embedding.weight.data, rank)
            self.down.weight.copy_(a)
            self.up.weight.copy_(b.T)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(input_ids))

    def extra_repr(self) -> str:
        return f'{self.num_embeddings}, {self.embedding_dim}, rank={self.rank}'


class FactorizedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 64):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.down = nn.Linear(self.in_features, rank, bias=False)
        self.up = nn.Linear(rank, self.out_features, bias=linear.bias is not None)

        with torch.no_grad():
            a, b = low_rank_factors(linear.weight.data, rank)
            self.up.weight.copy_(a)
            self.down.weight.copy_(b)
            if linear.bias is not None and self.up.bias is not None:
                self.up.bias.copy_(linear.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}'


def factorize_embeddings(model: PreTrainedModel, rank: int = 64) -> PreTrainedModel:
    embeddings = model.base_model.embeddings
    embeddings.word_embeddings = FactorizedEmbedding(embeddings.word_embeddings, rank=rank)  # ty:ignore[invalid-assignment,invalid-argument-type,unresolved-attribute]
    return model


def factorize_encoder_linears(model: PreTrainedModel, rank: int = 64) -> PreTrainedModel:
    encoder = model.base_model.encoder
    targets = [
        (name, module)
        for name, module in encoder.named_modules()  # ty:ignore[unresolved-attribute]
        if isinstance(module, nn.Linear)
        and rank * (module.in_features + module.out_features)
        < module.in_features * module.out_features
    ]
    for name, linear in targets:
        parent_name, _, child_name = name.rpartition('.')
        parent = encoder.get_submodule(parent_name) if parent_name else encoder  # ty:ignore[unresolved-attribute]
        setattr(parent, child_name, FactorizedLinear(linear, rank=rank))
    return model
