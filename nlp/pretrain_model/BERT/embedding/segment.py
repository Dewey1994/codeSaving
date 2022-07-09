import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super(SegmentEmbedding, self).__init__(vocab_size, embed_size, padding_idx=0)
        # https://github.com/huggingface/transformers/issues/15292
        # explain of padding_idx, 将那个padding的位置不算梯度