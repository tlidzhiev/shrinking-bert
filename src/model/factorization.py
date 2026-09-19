import torch
from torch import nn
from transformers import PreTrainedModel


class FactorizedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, rank: int = 64):
        super().__init__()
        vocab_size, hidden_size = embedding.weight.shape
        if not 0 < rank <= min(vocab_size, hidden_size):
            raise ValueError(f'rank must be in (0, {min(vocab_size, hidden_size)}], got {rank}')

        self.num_embeddings = vocab_size
        self.embedding_dim = hidden_size
        self.rank = rank
        self.down = nn.Embedding(vocab_size, rank)
        self.up = nn.Linear(rank, hidden_size, bias=False)

        dtype = embedding.weight.dtype
        with torch.no_grad():
            u, s, vh = torch.linalg.svd(embedding.weight.data.float(), full_matrices=False)
            sqrt_s = torch.sqrt(s[:rank])
            self.down.weight.copy_((u[:, :rank] * sqrt_s).to(dtype))
            self.up.weight.copy_((vh[:rank] * sqrt_s.unsqueeze(1)).T.to(dtype))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(input_ids))

    def extra_repr(self) -> str:
        return f'{self.num_embeddings}, {self.embedding_dim}, rank={self.rank}'


def factorize_embeddings(model: PreTrainedModel, rank: int = 64) -> PreTrainedModel:
    embeddings = model.base_model.embeddings
    embeddings.word_embeddings = FactorizedEmbedding(embeddings.word_embeddings, rank=rank)  # ty:ignore[invalid-assignment,invalid-argument-type,unresolved-attribute]
    return model
