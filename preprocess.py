import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def trim_string(x, first_n_words=200):
    x = x.split(maxsplit=first_n_words)
    x = ' '.join(x[:first_n_words])
    return x

des_path = 'sentence/'
data_need_path = des_path + 'data_need.csv'
data_novel_path = des_path + 'novel_sentence.xlsx'

train_test_ratio = 0.95

df_novel = pd.read_excel(data_novel_path, header=None)
df_novel['label'] = -1
df_novel.rename(columns={0: 'text'}, inplace=True)
df_novel['text'] = df_novel['text'].apply(trim_string)

data_need = pd.read_csv(data_need_path)
data_need['label'] = 1
data_need['text'] = data_need['text'].apply(trim_string)

df_train, df_test = train_test_split(data_need, train_size = train_test_ratio, random_state=1)
df_test_with_novel = pd.concat([df_test, df_novel], ignore_index=True, sort=False)

df_train.to_csv(des_path + 'train.csv', index=False)
df_test_with_novel.to_csv(des_path + 'test.csv', index=False)
