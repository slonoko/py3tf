#%%
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

X, y = datasets.load_breast_cancer(True)
le = LabelEncoder()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y,random_state=1)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1))
#pipe_lr.fit(X_train, y_train)
#y_pred = pipe_lr.predict(X_test)
#print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
param_range=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid2=[{'logisticregression__C': param_range}]
gs2 = GridSearchCV(estimator=pipe_lr, param_grid= param_grid2, scoring="accuracy", cv=10, n_jobs=-1)
gs2= gs2.fit(X_train, y_train)
print(gs2.best_score_)
print(gs2.best_params_)
#%%
import numpy as np 
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)
scores = []

for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))

#%%
print('\nCV accuracy: %.3f +/- %.3f' %
      (np.mean(scores), np.std(scores)))

#%%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=-1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
      np.std(scores)))

#%%
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))
param_range=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid= [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': param_range,
                   'svc__gamma': param_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid= param_grid, scoring="accuracy", cv=10, n_jobs=-1)
gs= gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

#%%
