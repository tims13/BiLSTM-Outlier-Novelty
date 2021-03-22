import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def trim_string(x, first_n_words=200):
    x = x.split(maxsplit=first_n_words)
    x = ' '.join(x[:first_n_words])
    return x

des_path = 'sentence/'
data_path = des_path + 'annotation_sentence.xlsx'
data_pos_csv_path = des_path + 'data_pos.csv'
data_neg_csv_path = des_path + 'data_neg.csv'
data_irre_csv_path = des_path + 'data_irre.csv'
data_novel_csv_path = des_path + 'data_novel.csv'

train_test_ratio = 0.90

data_paths = [
    data_pos_csv_path,
    data_neg_csv_path,
    data_irre_csv_path,
    data_novel_csv_path
]

headers = [
    ['pos'],
    ['neg'],
    ['irre'],
    ['novel']
]

data = pd.read_excel(data_path)
data = data.iloc[:, 0:4]

for i in range(4):
    data_t = data.iloc[:, i].astype(str)
    data_t = data_t[data_t != 'nan']
    data_t.reset_index(drop=True, inplace=True)
    data_t.to_csv(data_paths[i], header=headers[i], index=0)

data_pos = pd.read_csv(data_pos_csv_path)
data_pos['label'] = 1
data_pos.rename(columns={'pos': 'text'}, inplace=True)
data_pos['text'] = data_pos['text'].apply(trim_string)

data_neg = pd.read_csv(data_neg_csv_path)
data_neg['label'] = 0
data_neg.rename(columns={'neg': 'text'}, inplace=True)
data_neg['text'] = data_neg['text'].apply(trim_string)

# preprocess unknow intent / novelty
data_novel = pd.read_csv(data_novel_csv_path)
data_novel['label'] = 2
data_novel.rename(columns={'novel': 'text'}, inplace=True)
data_novel['text'] = data_novel['text'].apply(trim_string)

df_train, df_test = train_test_split(data_neg, train_size = train_test_ratio, random_state=1)
df_test_with_novel = pd.concat([df_test, data_novel], ignore_index=True, sort=False)

df_train.to_csv(des_path + 'train.csv', index=False)
df_test_with_novel.to_csv(des_path + 'test.csv', index=False)
