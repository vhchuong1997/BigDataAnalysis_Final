import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import time
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import RandomizedSearchCV
import xgboost
import lightgbm as lgb
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
import time
# Deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from seaborn import load_dataset
from sklearn import svm
# from thundersvm import SVC


train = pd.read_csv('./copy_final_train2.csv')
test = pd.read_csv('./copy_final_test2.csv')
print('Data loaded')

df1 = train.drop(['target','registration_init_time', 'expiration_date'], axis=1)
df2 = test.drop(['id', 'registration_init_time', 'expiration_date'], axis=1)

df = pd.concat([df1, df2], axis=0)

cat_list = ['msno', 'song_id', 'source_system_tab', 'source_screen_name', \
'source_type', 'gender', 'name', 'artist_name', 'country_code', 'registration_code']

# df2 = df2.dropna()

df[cat_list] = df[cat_list].apply(LabelEncoder().fit_transform)

from sklearn import preprocessing

x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
arr_scaled = min_max_scaler.fit_transform(df) 

df = pd.DataFrame(arr_scaled, columns=df.columns,index=df.index)
# df = pd.DataFrame(min_max_scaler.fit_transform(df.T), columns=df.columns, index=df.index)

df1 = df.iloc[0:7377418,:]
df2 = df.iloc[7377418:,:]

from sklearn.model_selection import train_test_split
X = df1
y = train['target'].values
# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, shuffle=False)# random_state=42)
X_train, y_train = X, y
X_test = df2
print('Finished preprocessing data')

print('Beginning to train SVM')
clf_svm = svm.SVC()
clf_svm.fit(X_train, y_train)

y_pred = clf_svm.predict(X_test)
# y_pred = clf_svm.predict(X_test)
submission = pd.read_csv('sample_submission.csv')
submission['target'] = y_pred
submission.to_csv('submission_svm.csv', index= False)