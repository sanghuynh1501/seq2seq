import torch
from torch import nn, Tensor
import torch.nn.functional as F

from encoder import Encoder


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int,
                 encoder: Encoder,
                 method="dot"):
        super().__init__()

        self.method = method
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.bidirectional = encoder.bidirectional
        print("self.bidirectional ", self.bidirectional)

        if self.bidirectional:
            self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        else:
            self.attn_in = enc_hid_dim + dec_hid_dim

        if self.method == "dot":
            pass

        elif self.method == "concat":
            self.attn = nn.Linear(self.attn_in, attn_dim)

        elif self.method == "general":
            if self.bidirectional:
                self.attn = nn.Linear(dec_hid_dim, dec_hid_dim * 2)
            else:
                self.attn = nn.Linear(dec_hid_dim, dec_hid_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        if self.method == "dot":
            encoder_outputs = encoder_outputs.permute(1, 0, 2)

            decoder_hidden = decoder_hidden.unsqueeze(-1)

            if self.bidirectional:
                decoder_hidden = decoder_hidden.repeat(1, 2, 1)

            energy = encoder_outputs.bmm(decoder_hidden).squeeze(-1)

            return F.softmax(energy, dim=1)

        elif self.method == "concat":
            src_len = encoder_outputs.shape[0]

            repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

            encoder_outputs = encoder_outputs.permute(1, 0, 2)

            energy = torch.tanh(self.attn(torch.cat((
                repeated_decoder_hidden,
                encoder_outputs),
                dim=2)))

            energy = torch.sum(energy, dim=2)

            return F.softmax(energy, dim=1)

        elif self.method == "general":
            decoder_hidden = self.attn(decoder_hidden).unsqueeze(-1)

            encoder_outputs = encoder_outputs.permute(1, 0, 2)

            energy = encoder_outputs.bmm(decoder_hidden).squeeze(-1)

            return F.softmax(energy, dim=1)