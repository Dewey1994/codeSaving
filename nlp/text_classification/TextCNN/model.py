import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, embed_num, embed_dim, label_num, kernel_num, kernel_sizes, dropout):
        super(TextCNN, self).__init__()
        ci = 1
        co = kernel_num
        self.embed = nn.Embedding(embed_num, embed_dim)
        self.conv = nn.ModuleList([nn.Conv2d(ci, co, (f, embed_dim), padding=(2, 0)) for f in kernel_sizes])
        self.dropout = nn.Dropout(p=dropout)
        self.pool = nn.AvgPool1d(embed_dim)
        self.fc = nn.Linear(co*len(kernel_sizes), label_num)

    def forward(self, x):
        x = self.embed(x).unsqueeze(1)
        # x-shape: [Batch_size,1,seq_length, embed_dim]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]
        # channel 从1变成kernel_sizes，对后两维进行卷积，kernel-shape为(设定的kernel_size，embed_dim)
        # 对embed_dim直接压缩到1维，对seq_length则模拟n-gram操作
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = self.dropout(torch.cat(x,1))
        logits = self.fc(x)
        return logits


if __name__ == '__main__':
    a = torch.ones((32, 20), dtype=torch.long)
    model = TextCNN(3000, 256, 20, 3,[3,5,7],0.3)
    b = model(a)
    print(b)
