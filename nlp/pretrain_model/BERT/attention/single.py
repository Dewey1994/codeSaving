import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        # query[batch,multi_heads,seq_len,hidden_dim] [batch,12,seq_len,64]
        # key[batch,multi_heads,seq_len,hidden_dim] [batch,12,seq_len,64]
        # value[batch,multi_heads,seq_len,hidden_dim] [batch,12,seq_len,64]
        # mask[batch,1,1,seq_len]
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.size(-1))
        # score [batch,multi_heads,seq_len,seq_len] [batch,12,seq,seq]
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(score, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


if __name__ == '__main__':
    model = Attention()
    query = torch.randn((32,12,50,64))
    key = torch.randn((32,12,50,64))
    value = torch.randn((32,12,50,64))
    mask = torch.zeros((32,1,1,50))
    mask[:,:,:,:20]=1
    res = model(query,key,value,mask)