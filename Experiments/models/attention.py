import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, query_dim=256):
        super().__init__()
        self.scale = 1. / np.sqrt(query_dim)

    # dot product attention
    def forward(self, query, keys, values):
        # Query = (batch_size, emb_dim)
        # Keys = Values = (batch_size, #frames, emb_dim)
        query = query.unsqueeze(1)  # (batch_size, emb_dim) -> (batch_size, 1, emb_dim)
        keys = keys.transpose(1, 2)  # (batch_size, #frames, emb_dim)  -> (batch_size, emb_dim, #frames)
        energy = torch.bmm(query, keys)  # (batch_size, 1, #frames)
        energy = F.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize
        att_weights = torch.bmm(energy, values).squeeze(1)  # (batch_size, emb_dim)
        return att_weights