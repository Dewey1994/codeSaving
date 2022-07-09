import torch.nn as nn
from .single import Attention


class MultiHeadAttention(nn.Module):
    def __init__(self, head, hidden_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % head == 0
        self.d_k = hidden_size // head
        self.head = head
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.output_layers = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

        self.attention = Attention()

    def forward(self, query, key, value, mask, dropout):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.head, self.d_k).permute(0, 2, 1, 3) for l, x in
                             zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(query, key, value, mask, self.dropout)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.head * self.d_k)
        return self.output_layers(x)