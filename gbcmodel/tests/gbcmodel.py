import os
# import sys
# sys.path.append(".")
import unittest
from unittest import mock, TestCase, main, skip
import pandas as pd
import numpy as np
from ..gbcmodel import GBCmodel


path = os.path.dirname(__file__)

class UpdateTestCase(TestCase):
    X = None
    y = None
    def setUp(self):
        super().setUp()
        self.load_data(path+"/train.csv")

    def load_data(self,filepath, target_field="is_bad", id_field = "Id"):
        df = pd.read_csv(filepath)
        self.y = df[target_field].values
        cols = [c for c in df.columns if c not in [target_field, id_field]]
        self.X = df[cols]


class TestGBCmodel(UpdateTestCase):

    def test_data_size(self):
        rows = 0
        cols = 0
        self.assertEqual(self.X.shape[0], 10000)
        self.assertEqual(self.X.shape[1], 22)

    def test_preprocess_category(self):
        # Check  handle new category levels at prediction(test) time 
        clf = GBCmodel()
        Xtrain = self.X.iloc[0:10]
        #The column 'emp_length' in the train data frame
        actual1 = ['10', '1', '4', '10', '10', '4', '10', '6', '2', '1']
        self.assertEqual(Xtrain['emp_length'].tolist(), actual1)

        Xtest = self.X.iloc[100:110]
        # The column 'emp_length' in the test data frame
        actual2 = ['1', '3', '7', '2', '8', '8', '1', '1', '4', '10']
        self.assertEqual(Xtest['emp_length'].tolist(), actual2)

        #Convert string categories into numbers
        Xtr = clf.preprocess(Xtrain, test=False)
        actual_1 = [2.0, 1.0, 4.0, 2.0, 2.0, 4.0, 2.0, 5.0, 3.0, 1.0]
        self.assertEqual(Xtr[:,0].tolist(), actual_1)
        #The maps to convert categories:
        #  clf.mp = {'emp_length': {'1': 1, '10': 2, '2': 3, '4': 4, '6': 5}

        #The categories '3', '7', '8' are new in the test dataset.
        #We should replace them on zero.
        Xte = clf.preprocess(Xtest, test=True)
        actual_2 = [1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0, 1.0, 4.0, 2.0]
        self.assertEqual(Xte[:, 0].tolist(), actual_2)

    def test_preprocess_missing(self):
        # can handle missing values 
        clf = GBCmodel()
        Xn = self.X.iloc[50:55].copy()
        # Xn['mths_since_last_delinq'] [34.0, 4.0, nan, nan, 63.0]
        self.assertEqual(Xn['mths_since_last_delinq'].isnull().sum(), 2)
        Xne = clf.preprocess(Xn)
        self.assertEqual(np.sum(np.isnan(Xne[:,13])), 0)

    def test_fit_evaluate(self):
        clf = GBCmodel()
        clf.fit(self.X, self.y)
        ev = clf.evaluate(self.X, self.y)
        self.assertGreater(ev['f1_score'], 0.1)

    def test_fit_predict(self):
        clf = GBCmodel()
        Xt = self.X.iloc[0:8000]
        yt = self.y[0:8000]
        Xp = self.X.iloc[8000:]
        clf.fit(Xt, yt)
        pr = clf.predict(Xp)
        self.assertEqual(pr.shape[0], 2000)

    def test_fit_predict_proba(self):
        clf = GBCmodel()
        Xt = self.X.iloc[0:8000]
        yt = self.y[0:8000]
        Xp = self.X.iloc[8000:]
        clf.fit(Xt, yt)
        pr = clf.predict_proba(Xp)
        self.assertEqual(pr.shape[0], 2000)
        self.assertEqual(pr.shape[1], 2)

    def test_tune_parameters(self):
        clf = GBCmodel()
        Xt = self.X.iloc[0:10000]
        yt = self.y[0:10000]
        res = clf.tune_parameters(Xt, yt, n_iter=5)
        self.assertGreater(res['best_scores']['f1_score'], 0.2)

if __name__ == '__main__':
    unittest.main()