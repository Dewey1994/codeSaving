import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, num_sparse_features, num_dense_features, embedding_dim, hidden_units):
        super(DeepFM, self).__init__()

        # Embedding层
        self.embedding = nn.Embedding(num_sparse_features, embedding_dim)

        # FM部分
        self.fm_linear = nn.Linear(num_sparse_features + num_dense_features, 1)

        # Deep部分
        self.deep_layers = nn.ModuleList()
        num_layers = len(hidden_units)
        for i in range(num_layers):
            if i == 0:
                self.deep_layers.append(nn.Linear(embedding_dim + num_dense_features, hidden_units[i]))
            else:
                self.deep_layers.append(nn.Linear(hidden_units[i - 1], hidden_units[i]))

        self.deep_output = nn.Linear(hidden_units[-1], 1)

    def forward(self, sparse_inputs, dense_inputs):
        # 处理稀疏特征
        embedded_sparse_inputs = self.embedding(sparse_inputs)
        sparse_output = torch.sum(embedded_sparse_inputs, dim=1)

        # 处理稠密特征
        dense_output = dense_inputs

        # FM部分
        fm_input = torch.cat((sparse_output, dense_output), dim=1)
        fm_linear_output = self.fm_linear(fm_input).squeeze(dim=1)

        # 二阶特征交互
        fm_output = 0.5 * torch.sum(
            (torch.sum(embedded_sparse_inputs, dim=1)) ** 2 - torch.sum(embedded_sparse_inputs ** 2, dim=1), dim=1,
            keepdim=True)

        # Deep部分
        deep_input = torch.cat((embedded_sparse_inputs.view(embedded_sparse_inputs.size(0), -1), dense_output), dim=1)
        deep_output = deep_input
        for layer in self.deep_layers:
            deep_output = layer(deep_output)
            deep_output = nn.ReLU()(deep_output)
        deep_output = self.deep_output(deep_output)

        # 融合FM和Deep部分的输出
        output = fm_linear_output + fm_output + deep_output

        return torch.sigmoid(output)
