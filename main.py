import os
import numpy as np
import pandas as pd
from torchtext.data import Field
from torchtext.vocab import Vectors
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
from tqdm import tqdm
import spacy
from spacy.symbols import ORTH
from sklearn.neighbors import LocalOutlierFactor
from model import BiLSTM
from sklearn.svm import OneClassSVM

def l2_normalize(x, axis=-1):
    y = np.max(np.sum(x ** 2, axis, keepdims=True), axis, keepdims=True)
    return x / np.sqrt(y)

data_review_csv_path = 'laptop/amazon_reviews.csv'
data_outliers_csv_path = 'laptop/outliers.csv'
data_novelty_csv_path = 'laptop/novel_needs.csv'
glove_path = 'glove.6B.300d.txt'


my_tok = spacy.load('en')
my_tok.tokenizer.add_special_case('<eos>', [{ORTH: '<eos>'}])
my_tok.tokenizer.add_special_case('<bos>', [{ORTH: '<bos>'}])
my_tok.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])
def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]

TEXT = Field(sequential=True, tokenize=spacy_tok, lower=True)
tst_datafields = [
    ("comment_text", TEXT)
]
tst = TabularDataset(
    path = data_review_csv_path,
    format = 'csv',
    skip_header = True,
    fields = tst_datafields
)

novel_datafields = [
    ("novel", TEXT)
]
novel = TabularDataset(
    path = data_novelty_csv_path,
    format = 'csv',
    skip_header = True,
    fields = novel_datafields
)

cache = '.vector_cache'
vectors = Vectors(name=glove_path, cache=cache)
TEXT.build_vocab(tst, vectors=vectors)

data_iter = Iterator(tst, batch_size=1, device=-1, sort=False, sort_within_batch=False, repeat=False, shuffle=False)
novel_iter = Iterator(novel, batch_size=1, device=-1, sort=False, sort_within_batch=False, repeat=False, shuffle=False)

vocab = TEXT.vocab

vocab_size = len(vocab)
emb_dim = 300
hidden_dim = 64
emb_matrix = vocab.vectors
model = BiLSTM(vocab_size, hidden_dim, emb_dim, emb_matrix)

print("Computing the deep features...")
'''
features = []
for x in tqdm(data_iter):
    feature = model(x.comment_text)
    features.append(feature)

feats = []
for f in features:
    feats.append(f.detach().numpy())
feats = np.vstack(feats)
'''
novel_features = []
for x in tqdm(novel_iter):
    print(x.novel.shape)
    feature = model(x.novel)
    novel_features.append(feature)

novel_feats = []
for f in novel_features:
    novel_feats.append(f.detach().numpy())
novel_feats = np.vstack(novel_feats)
print(novel_feats.shape)

# feats = l2_normalize(feats, 0)

lof = LocalOutlierFactor(n_neighbors=20, contamination='auto', n_jobs=-1, novelty=True)
res = lof.fit(feats)

# find the outliers of the dataset
'''
data_review = pd.read_csv(data_review_csv_path)
outliers = np.where(res == -1)
ngf = lof.negative_outlier_factor_
data_outliers = data_review.iloc[np.argsort(ngf)[0:outliers[0].shape[0]]]
data_outliers.to_csv(data_outliers_csv_path, header=1, index=0)
print("The outliers are saved!")
'''

# novelty
res = lof.predict(novel_feats)
scores = lof.score_samples(novel_feats)
print("results:")
print(res)
print(scores)


clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(feats)
res = clf.predict(novel_feats)
scores = clf.score_samples(novel_feats)
print("results:")
print(res)
print(scores)

