import os
import numpy as np
import pandas as pd

data_review_path = 'laptop/amazon_reviews - IEEE.xlsx'
data_review_csv_path = 'laptop/amazon_reviews.csv'

print("Preprocess...")
data = pd.read_excel(data_review_path, index_col=0, header=1)

data_review = data.iloc[:,0]
data_review = data_review[data_review != ' ']

for index in range(1, data.shape[1]):
    temp = data.iloc[:,index]
    temp = temp[temp != ' ']
    data_review = pd.concat([data_review, temp])

data_review.reset_index(drop=True, inplace=True)

data_review.to_csv(data_review_csv_path, header=1, index=0)

print("The results are save in amazon_reviews.csv!")
