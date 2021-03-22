import os
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

data_outliers_csv_path = 'laptop/outlier.csv'
data_review_csv_path = 'laptop/amazon_reviews.csv'
data_np_path = 'laptop/review_novel'

data = np.load(data_np_path + '.npz')
review = data['review']
novel = data['novel']

print(review.shape)
print(novel.shape)

lof = LocalOutlierFactor(n_neighbors=20, contamination='auto', n_jobs=-1, novelty=True)
res = lof.fit(review)

data_review = pd.read_csv(data_review_csv_path)
ngf = lof.negative_outlier_factor_
data_outliers = data_review.iloc[np.argsort(ngf)[0:10]]
data_outliers.to_csv(data_outliers_csv_path, header=1, index=0)

print(np.sort(ngf))

res = lof.predict(novel)
scores = lof.score_samples(novel)
print("LOF results:")
print(res)
print(scores)

clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(review)
res = clf.predict(novel)
scores = clf.score_samples(novel)
print("oc-SVM results:")
print(res)
print(scores)