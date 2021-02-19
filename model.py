import torch.nn as nn
import torch.nn.functional as F
from transformers import  BertModel, BertConfig

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


class BERT(nn.Module):
    def __init__(self, feature_len):
        super(BERT, self).__init__()
        options_name = "bert-base-uncased"
        config = BertConfig.from_pretrained(options_name)
        self.encoder = BertModel.from_pretrained(options_name, config=config)
        embedding_dim = self.encoder.config.hidden_size
        self.fc = nn.Linear(embedding_dim, feature_len)

    def forward(self, text):
        last_hidden_states = self.encoder(text)
        text_embeddings = last_hidden_states[0][:,0,:]
        text_features = self.fc(text_embeddings)
        # text_features = self.tanh(features)
        return text_features
