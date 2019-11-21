from typing import Tuple

import torch
from torch import nn, Tensor


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 bidirectional=False,
                 dropout=0.5):
        super().__init__()

        self.dropout = dropout
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=bidirectional)

        if bidirectional:
            self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        else:
            self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        if self.bidirectional:
            hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        else:
            hidden = torch.tanh(self.fc(hidden)).squeeze(0)

        return outputs, hidden


encoder = Encoder(100, 32, 64, 64, False)
inputs = torch.ones(23, 128).long()
outputs, hidden = encoder(inputs)
print("outputs.shape ", outputs.shape)
print("hidden.shape ", hidden.shape)
