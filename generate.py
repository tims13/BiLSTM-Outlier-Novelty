import os
import numpy as np
import pandas as pd
import torch
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import Iterator
from tqdm import tqdm
from transformers import BertTokenizer
from model import BERT

data_np_path = 'laptop/review_novel'
data_review_csv_path = 'laptop/amazon_reviews.csv'
data_novelty_csv_path = 'laptop/novel_needs.csv'

feature_len = 64

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# load data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model parameter
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
rev_field = [('comment_text', text_field)]
nov_field = [('novel', text_field)]

review = TabularDataset(
    path = data_review_csv_path,
    format = 'csv',
    skip_header = True,
    fields = rev_field
)

novel = TabularDataset(
    path = data_novelty_csv_path,
    format = 'csv',
    skip_header = True,
    fields = nov_field
)

review_iter = Iterator(review, batch_size=1, device=device, sort=False, sort_within_batch=False, repeat=False, shuffle=False)
novel_iter = Iterator(novel, batch_size=1, device=device, sort=False, sort_within_batch=False, repeat=False, shuffle=False)

model = BERT(feature_len).to(device)

print("Computing deep features...")

review_features = []
for x in tqdm(review_iter):
    text = x.comment_text.type(torch.LongTensor)
    text = text.to(device)
    feature = model(text)
    review_features.append(feature.detach().numpy())
review_features = np.vstack(review_features)
print(review_features.shape)

novel_features = []
for x in tqdm(review_iter):
    text = x.novel.type(torch.LongTensor)
    text = text.to(device)
    feature = model(text)
    novel_features.append(feature.detach().numpy())
novel_features = np.vstack(novel_features)
print(novel_features.shape)

# save the results
np.savez(data_np_path, review=review_features, novel=novel_features)
print("The features are saved in "+ data_np_path)
