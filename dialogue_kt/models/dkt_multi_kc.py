"""
DKT model that averages KC embeddings per item
Based on pykt implementation: https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/dkt.py
"""

import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

from dialogue_kt.utils import device

class DKTMultiKC(Module):
    def __init__(self, num_kcs: int, emb_size: int, dropout=0.1):
        super().__init__()
        self.num_kcs = num_kcs
        self.emb_size = emb_size
        self.hidden_size = emb_size

        self.interaction_emb = Embedding(self.num_kcs + 2, self.emb_size)

        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size, self.num_kcs)

        print("trainable params:", sum([param.numel() for param in self.parameters()]))

    def forward(self, batch):
        batch_size, max_seq_len, max_num_kcs = batch["kc_ids"].shape

        # Get KC embeddings
        xemb = self.interaction_emb(batch["kc_ids"]) # B x L x K x D
        # Mask out padded KC ids
        kc_pad_mask = torch.arange(max_num_kcs).repeat(batch_size, max_seq_len, 1).to(device) >= batch["num_kcs"].unsqueeze(2)
        zeros = torch.zeros_like(xemb).to(device)
        xemb = torch.masked_scatter(xemb, kc_pad_mask.unsqueeze(3), zeros)
        # Average KC embeddings
        xemb = xemb.sum(dim=2) / batch["num_kcs"].unsqueeze(2) # B x L x D
        # Add correctness embeddings, clip is so padding labels don't go out of range
        correct_emb = self.interaction_emb(self.num_kcs + torch.clip(batch["labels"], min=0)) # B x L x D
        xemb += correct_emb

        # Run embeddings through LSTM to get output predictions
        h, _ = self.lstm_layer(xemb)
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y) # B x L x K(all)

        return y
