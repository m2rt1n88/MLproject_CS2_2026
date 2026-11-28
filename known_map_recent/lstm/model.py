"""Shared-encoder Siamese LSTM. logit = g(team1) - g(team2), so swapping sides gives 1 - p."""
import torch
from torch import nn


class MapLSTM(nn.Module):
    def __init__(self, n_maps, n_static, n_seq, hidden=32, emb=8, mlp=32, dropout=0.1):
        super().__init__()
        self.map_emb = nn.Embedding(n_maps, emb)          # index 0 = unknown map
        self.lstm = nn.LSTM(n_seq + emb, hidden, batch_first=True)
        self.g = nn.Sequential(nn.Linear(hidden + n_static + emb, mlp), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(mlp, 1))

    def encode(self, seq, seq_maps, lengths):
        """seq [B,T,F] left aligned; zero-length sequences produce a zero hidden state."""
        x = torch.cat([seq, self.map_emb(seq_maps)], -1)
        out, _ = self.lstm(x)
        mask = torch.arange(seq.shape[1], device=seq.device)[None, :] < lengths[:, None]
        last = (lengths - 1).clamp(min=0)
        h = out[torch.arange(len(seq)), last]
        return h * mask.any(1, keepdim=True).float()

    def side(self, static, seq, seq_maps, lengths, target):
        h = self.encode(seq, seq_maps, lengths)
        return self.g(torch.cat([h, static, self.map_emb(target)], -1)).squeeze(-1)

    def forward(self, static, seq, seq_maps, lengths, target):
        """static [B,2,S], seq [B,2,T,F], seq_maps [B,2,T], lengths [B,2], target [B] -> logit [B]."""
        b = len(target)
        g = self.side(static.flatten(0, 1), seq.flatten(0, 1), seq_maps.flatten(0, 1), lengths.flatten(),
                      target.repeat_interleave(2)).view(b, 2)
        return g[:, 0] - g[:, 1]


def make_batch(static, seq, seq_maps, lengths, target):
    """Numpy arrays -> tensors in the order forward() expects."""
    return (torch.as_tensor(static, dtype=torch.float32), torch.as_tensor(seq, dtype=torch.float32),
            torch.as_tensor(seq_maps, dtype=torch.long), torch.as_tensor(lengths, dtype=torch.long),
            torch.as_tensor(target, dtype=torch.long))


@torch.no_grad()
def predict_proba(model, batch, size=2048):
    model.eval()
    n = len(batch[-1])
    out = [torch.sigmoid(model(*(t[i:i + size] for t in batch))) for i in range(0, n, size)]
    return torch.cat(out).numpy() if out else torch.zeros(0).numpy()
