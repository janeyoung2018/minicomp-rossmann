## Rossman Kaggle Mini-Competition

## Setup

1. Checkout the repo

2. Start a Jupyter Notebook in the repository

3. Import packages
```bash
import os
import pandas as pd
```
3. Unzip data
```bash
!python data.py
```
4. Load raw_data
```bash
home = os.path.join(os.environ['HOME'], 'MiniChallenge', 'data')

def load_raw_data(nrows=None):
    raw_data = {}
    for fi in os.listdir(home):
        if 'csv' in fi:
            print(fi)
            raw_data[fi] = pd.read_csv(os.path.join(home, fi))
    return raw_data 

raw_data = load_raw_data()
```
5. Get train and store data
```bash
train = raw_data['train.csv']
sotre = raw_data['store.csv']
```
6. Merge store and train:
```bash
newdf = train.merge(store,how='left',on='Store')
```

7. Split train and test data
```bash
```
8. Clean data:
```bash
from cleandata import cleanData
newdf = cleanData(newdf)
```
9. Run model
```bash
```
10. Test on test.csv
```bash
```
5. Run the model:
```bash
TBD
```
