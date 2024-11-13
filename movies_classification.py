# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 08:59:19 2024

@author: rpcha
"""

import pandas as pd
df = pd.read_csv('C:/Data_Science_2.0/RandomForest/movies_classification.csv')
df.info()

# Build a new column based on these give columns
df = pd.get_dummies(df, columns=['3D_available', 'Genre'], drop_first=True)

predictors=df.loc[:, df.columns!="Start_Tech_Oscar"]

target = df["Start_Tech_Oscar"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
rand_for = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

# n_estimators is number of trees in the forest, always in range 500 to 1000
# n_jobs=1 means number of jobs running parallel 
# if it is -1 then multiple jobs running parallel
# random_state controls randomness in bootstrapping
# Bootstrapping is getting samples replaced

rand_for.fit(X_train, y_train)
pred_X_train = rand_for.predict(X_train)
pred_X_test = rand_for.predict(X_test)

# Performance matrics
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(pred_X_test, y_test)
confusion_matrix(pred_X_test, y_test)

