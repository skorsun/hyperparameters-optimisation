import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler, Normalizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  cross_val_predict, cross_val_score
from sklearn.metrics import f1_score, log_loss
from hyperopt import fmin, tpe, hp, Trials
pd.options.mode.chained_assignment = None

class GBCmodel:
    """Gradient Boosting Classifier
    Parameters
    ----------

    Examples
    --------
    """
    def __init__(self):
        self.le = None
        self.catcols = None
        self.numcols = None
        self.clf =  GradientBoostingClassifier()
        self.mp = {}

    def preprocess(self, df, test=False):
        df.fillna(0, inplace=True)

        if not test:
            self.catcols = []
            self.numcols = []
            self.mp = {}
            for k in df.columns:
                if df[k].dtype is np.dtype('float64') or df[k].dtype is np.dtype('int64'):
                    self.numcols.append(k)
                else:
                    self.catcols.append(k)
                    uniq = df[k].unique()
                    d = {u: e+1 for e, u in enumerate(sorted(uniq))}
                    self.mp[k] = d
        df[self.catcols] = df[self.catcols].astype("str")
        df[self.numcols] = df[self.numcols].apply(lambda x: 0 if type(x) is "str" else x)
      #  print(self.mp)
        Xn = df[self.numcols].values

        for c in self.catcols:
            map = self.mp[c]
            df[c] = df[c].map(self.mp[c]).fillna(0).astype("int")
        Xc = df[self.catcols].values
        if not test:
            self.scaler = RobustScaler()
            Xn = self.scaler.fit_transform(Xn)
        else:
            Xn = self.scaler.transform(Xn)
        X = np.hstack([Xc,Xn])
        return X


    def fit(self, X, y, params=None):
        Xt = self.preprocess(X, False)
        self.clf.fit(Xt, y)

    def predict(self,X):
        Xt = self.preprocess(X, True)
        pr = self.clf.predict(Xt)
        return pr

    def predict_proba(self,X):
        Xt = self.preprocess(X, True)
        pr = self.clf.predict_proba(Xt)
        return pr

    def evaluate(self, X, y):
        pr = self.predict(X)
        pp = self.predict_proba(X)
        return {'f1_score': f1_score(y, pr), 'log_loss': log_loss(y, pp)}

    def tune_parameters(self, X, y, n_iter=10 ):
        Xt = self.preprocess(X, False)
        train_data, test_data, train_targets,  test_targets = train_test_split(
            Xt, y, test_size=0.2, random_state=42, stratify=y)
        # possible values of parameters
        space = {'n_estimators': hp.quniform('n_estimators', 10, 200, 1),
                 'max_depth': hp.quniform('max_depth', 2, 20, 1),
                 'learning_rate': hp.loguniform('learning_rate', -5, 0)
                 }

        def gb_f1_cv(params):
            # the function gets a set of variable parameters in "param"
            paramsg = {'n_estimators': int(params['n_estimators']),
                      'max_depth': int(params['max_depth']),
                      'learning_rate': params['learning_rate']}

            # we use this params to create a new GradientBoostingClassifier
            model = GradientBoostingClassifier(random_state=42, n_estimators=int(paramsg['n_estimators']),
                                         max_depth=int(paramsg['max_depth']), learning_rate=paramsg['learning_rate'])

            # and then conduct the cross validation with the same folds as before
            score = -cross_val_score(model, train_data, train_targets, cv=5, scoring="f1", n_jobs=-1).mean()
            return score

        # trials will contain logging information
        trials = Trials()

        best = fmin(fn=gb_f1_cv,  # function to optimize
                    space=space,
                    algo=tpe.suggest,  # optimization algorithm, hyperotp will select its parameters automatically
                    max_evals=n_iter,  # maximum number of iterations
                    trials=trials,  # logging
                    rstate=np.random.RandomState(2019)  # fixing random state for the reproducibility
                    )
        self.clf = GradientBoostingClassifier(random_state=42, n_estimators=int(best['n_estimators']),
                                           max_depth=int(best['max_depth']), learning_rate=best['learning_rate'])
        self.clf.fit(train_data, train_targets)
        tpe_test_score = f1_score(test_targets, self.clf.predict(test_data))
        logloss = log_loss(test_targets, self.clf.predict_proba(test_data))
        return {'best_parametrs': best, 'best_scores': {'f1_score': tpe_test_score, 'logloss': logloss}}

