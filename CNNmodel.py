import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        embedded = self.embedding(text)
        # embedded = [sent len, batch size, emb dim]
        embedded = embedded.permute(1, 2, 0)
        # embedded = [batch size, emb dim, sent len]
        conved = self.conv(embedded)
        # conved = [batch size, hidden dim, sent len - filter_size + 1]
        pooled = F.max_pool1d(conved, conved.shape[2])
        pooled = pooled.squeeze(2)
        # pooled = [batch size, hidden dim]
        return self.fc(pooled)





