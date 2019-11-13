# Introduction

## Gradient Boosting Classifier (GBCmodel)
One of the important task of any Data Scientist  is finding the best predictive model for users’ data, making  lots of data science decisions automatically along the way.
For example you need to create a new binary classification model for different datasets with numeric and categorial features.
You can find here Gradient Boosting Classifier with hyperparameters tunning. 


## Installations
python setup.py install

## Dependencies

- Python 3.6
- scikit-learn==0.21.3
- Other required packages are summarized in `requirements.txt`.

# Quick start
Create a file classify.py

```
import numpy as np
import os
import pandas as pd
from gbcmodel.gbcmodel import GBCmodel
path = os.path.dirname(__file__)

def main(filetrain, target_field="is_bad", id_field = "Id"):
    df = pd.read_csv(filetrain)
    y = df[target_field].values
    cols = [c for c in df.columns if c not in [target_field, id_field]]
    X = df[cols]
    clf = GBCmodel()
    res = clf.tune_parameters(X, y, n_iter=10)
    print(res)
    ev = clf.evaluate(X, y)
    print(ev)

if __name__ == '__main__':
    main(path + "/train.csv")

```

This script trains the model for 10 steps and classify test data with the best parameters of Gradient Boosting Classifier .

## Methods: 
Input features are given as a pandas DataFrame with a mix of numeric and categorical data. <br>
*Example* : <br>
 X = pd.DataFrame({'feat1': ['a', 'b', 'a'], 'feat2': [1, 2, 3]})<br>
 y = np.array([0, 0, 1]) <br>
 
### 1. Fit on training data:
**self.fit(X, y)**  
*Input  X* : pd.DataFrame  -  features <br>  
*Input features y* : np.ndarray - Ground truth labels as a numpy array of 0-s and 1-s. <br>
*Output Type* : -  None <br>

### 2. Predict class labels on new data: 
**self.predict(X)** 
*Input  X* : pd.DataFrame  - Input features <br> 
*Output Type* :  np.ndarray <br>
*Output Example* :  np.array([1, 0, 1]) <br>
 
### 3. Predict the probability of each label: 
**self.predict_proba(X)**<br> 
*Input  X* : pd.DataFrame  -    Input features <br> 
*Output Type* : np.ndarray <br> 
*Output Example* :  np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])<br>
### 4. Get the value of the following metrics: F1-score, LogLoss: 
**self.evaluate(X, y)**<br>
*Input  X* : pd.DataFrame Input features <br>
*y* : np.ndarray Ground truth labels as a numpy array of 0-s and 1-s.<br>
*Output* :  Type  dict<br>
*Output Example* :  {'f1_score': 0.3, 'logloss': 0.7}<br>
 
### 5. Find the best hyperparameters using K-Fold cross-validation for evaluation: 
**self.tune_parameters(X, y)**<br> 
*Input  X* : pd.DataFrame     Input features <br> 
*y* : np.ndarray    Ground truth labels as a numpy array of 0-s and 1-s.<br> 
*Output Type* :  dict 

## Testing
You can use the dataset gbcmodel/tests/train.csv to test this class.<br>
It contains a binary classification  target "is_bad" for predicting loan defaults.  

  

  