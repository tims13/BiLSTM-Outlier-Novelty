import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, emb_dim, emb_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(emb_matrix) # load the pre_train embedding
        self.embedding.weight.requires_grad = False # embedding is non-trainable
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=2, dropout=0, bidirectional=True)

    def forward(self, seq):
        emb = self.embedding(seq)
        hdn, _ = self.encoder(emb)
        feature = hdn[-1,:,:] # take the last timestamp of the encoder output
        return feature
