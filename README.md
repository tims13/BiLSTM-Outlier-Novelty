# BiLSTM-Outlier-Novelty

1. Glove is needed.
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

2. Preprocess
```
python preprocess.py
```

3. Generate features and save
```
python generate.py
```

4. Find the novel samples
```
python novel.py
```