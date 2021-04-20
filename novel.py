import os
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import recall_score, precision_score

def evaluate_novel(c, train_features, test_features, y_true, name):
    model = LocalOutlierFactor(n_neighbors=20, contamination=c, novelty=True, n_jobs=-1)
    model.fit(train_features)
    y_pred = model.predict(test_features)
    y_pred[y_pred != -1] = 0
    y_pred[y_pred == -1] = 1
    rec_score = recall_score(y_true, y_pred)
    prec_score = precision_score(y_true, y_pred)
    print("--------------------------------------------")
    print(name + str(c) + ' RECALL:' + str(rec_score))
    print(name + str(c) + ' PRECISION:' + str(prec_score))
    print("--------------------------------------------")
    return y_pred

des_path = 'sentence/'
data_test_path = des_path + 'test.csv'
data_np_path = des_path + 'train_test'
data_result_csv_path = des_path + 'results.csv'

data_test = pd.read_csv(data_test_path)
y_true = data_test['label']
y_true = np.array(y_true, np.int)
data_test['y_true'] = y_true

data = np.load(data_np_path + '.npz')
train = data['train']
test = data['test']

print(train.shape)
print(test.shape)

print('LOF training...')
list_c = [0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2]
for c in list_c:
    y_pred = evaluate_novel(c, train, test, y_true, 'LOF')
    data_test[str(c)] = y_pred

'''
print('Isolation Forest training...')
isolation_forest = IsolationForest(random_state=0, contamination=0.5, n_jobs=-1)
y_pred = evaluate_novel(isolation_forest, train, test, y_true, 'ISO')
data_test['Iso'] = y_pred

print('OC-SVM training...')
oc_svm = OneClassSVM(gamma='auto')
y_pred = evaluate_novel(oc_svm, train, test, y_true, 'OC-SVM')
data_test['OC-SVM'] = y_pred
'''
data_test.to_csv(data_result_csv_path, index=False)
