import pandas as pd
import numpy as np
import sys
import sklearn.datasets as datasets
from sklearn import linear_model
from sklearn import model_selection
from matplotlib import pyplot as plt
from sklearn import tree
import csv
from sklearn import naive_bayes
import pickle
from sklearn import preprocessing
from sklearn.decomposition import PCA
import itertools
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn import model_selection
from sklearn.model_selection import cross_val_score


s=pd.read_csv(sys.argv[2],header=None)
a=pd.read_csv(sys.argv[3],header=None)


#training data
s_category=s.iloc[:,[1,3,5,6,7,8,9,13]]
s_continuous=s.iloc[:,[0,2,4,10,11,12,14]]
train_col_cat=[]

for i in range(len(s_category)):
    train_col_cat.append(list(s_category.iloc[i,0:]))

enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
enc.fit(train_col_cat)
train_col_cat = enc.transform(train_col_cat).toarray()
train_col_cat=[list(train_col_cat[i]) for i in range(len(train_col_cat))]

train_col_cont=[]
train_ans=[]
for i in range(len(s_category)):
    train_col_cont.append(list(s_continuous.iloc[i,0:-1]))
    train_ans.append(s_continuous.iloc[i,-1])
train_col=[train_col_cat[i]+train_col_cont[i] for i in range(len(train_col_cont))]


#testing data

a_category=a.iloc[:,[1,3,5,6,7,8,9,13]]
a_continuous=a.iloc[:,[0,2,4,10,11,12]]
test_col_cat=[]

for i in range(len(a_category)):
    test_col_cat.append(list(a_category.iloc[i,0:]))

# enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
# enc.fit(train_col_cat)
test_col_cat = enc.transform(test_col_cat).toarray()
test_col_cat=[list(test_col_cat[i]) for i in range(len(test_col_cat))]

test_col_cont=[]
# train_ans=[]
for i in range(len(a_category)):
    test_col_cont.append(list(a_continuous.iloc[i,0:]))
    # test_ans.append(s_continuous.iloc[i,-1])
test_col=[test_col_cat[i]+test_col_cont[i] for i in range(len(test_col_cont))]








x_train, x_test, y_train, y_test = model_selection.train_test_split(
  train_col, train_ans, test_size = 0.0, shuffle=False)


dt = tree.DecisionTreeClassifier(criterion='gini',splitter='best',
                              max_depth=17,min_samples_split=0.0177,
                               max_features=None,max_leaf_nodes=None,
                               min_impurity_decrease=1e-9,random_state=0)
dt.fit(x_train, y_train)
score_train = dt.score(x_train, y_train)
print('training acc:',score_train)

# score_test = dt.score(x_test, y_test)
# print('testing acc:',score_test)

kfold = model_selection.KFold(n_splits=10, random_state=1)

scoring = 'accuracy'
results = model_selection.cross_val_score(dt, x_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


dt_pred=dt.predict(test_col)

# score_test = dt.score(x_test, y_test)
# print('testing acc:',score_test)

scores = model_selection.cross_val_score(dt, x_train, y_train, cv=10, scoring='accuracy')
print(scores, 'mean:',np.mean(scores))

par_smooth=0.00001
# for i in range(10):
NB1 = naive_bayes.GaussianNB(var_smoothing=par_smooth)
NB1.fit(x_train, y_train)
score_train = NB1.score(x_train, y_train)
print('Gaussian NB training acc:',score_train)
# score_test = NB1.score(x_test,y_test)
# print('Gaussian NB testing acc:',score_test)
# multi_pred=NB1.predict(test_col)


NB2 = naive_bayes.BernoulliNB(fit_prior=True,alpha=1)
NB2.fit(x_train, y_train)
score_train = NB2.score(x_train, y_train)
print('Bernoulli NB training acc:',score_train)
# score_test = NB2.score(x_test,y_test)
# print('Bernoulli NB testing acc:',score_test)


NB3 = naive_bayes.MultinomialNB(fit_prior=False,alpha=1e-9)
NB3.fit(x_train, y_train)
score_train = NB3.score(x_train, y_train)
print('Multinomial NB training acc:',score_train)
# score_test = NB3.score(x_test,y_test)
# print('Multinomial NB testing acc:',score_test)

o=open(sys.argv[4],'a')
for k in range(len(a)):
  o.write(str(dt_pred[k]))
  o.write("\n")
o.close()
