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


s=pd.read_csv(sys.argv[2],header=None)
a=pd.read_csv(sys.argv[3],header=None)


train_col=[]
train_ans=[]
for i in range(len(s)):
    train_col.append(list(s.iloc[i,0:-1]))
    train_ans.append(s.iloc[i,-1])
enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
enc.fit(train_col)
# input(enc.categories_)
# input(enc.get_feature_names())
train_col = enc.transform(train_col).toarray()


#parsing testing data 
test_col=[]
# test_ans=[]
for i in range(len(a)):
    test_col.append(list(a.iloc[i,0:22]))
    # test_ans.append(a.iloc[i,-1])
test_col = enc.transform(test_col).toarray()


x_train, x_test, y_train, y_test = model_selection.train_test_split(
  train_col, train_ans, test_size = 0.0, shuffle=False)


dt = tree.DecisionTreeClassifier(criterion='gini',splitter='best',
                              max_depth=None,min_samples_split=2,
                               max_features=None,max_leaf_nodes=None,
                               min_impurity_decrease=0.0,random_state=0)
dt.fit(x_train, y_train)
score_train = dt.score(x_train, y_train)
print('training acc:',score_train)

# score_test = dt.score(test_col, test_ans)
# print('testing acc:',score_test)

dt_pred=dt.predict(test_col)

par_smooth=0.0002
NB1 = naive_bayes.GaussianNB(var_smoothing=par_smooth)
NB1.fit(x_train, y_train)
score_train = NB1.score(x_train, y_train)
print('Gaussian NB training acc:',score_train)
multi_pred=NB1.predict(test_col)
# score_test = NB1.score(test_col, test_ans)
# print('Gaussian NB testing acc:',score_test)
# par_smooth=par_smooth+0.0001

NB2 = naive_bayes.BernoulliNB()
NB2.fit(x_train, y_train)
score_train = NB2.score(x_train, y_train)
print('Bernoulli NB training acc:',score_train)
# score_test = NB2.score(test_col, test_ans)
# print('Bernoulli NB testing acc:',score_test)


NB3 = naive_bayes.MultinomialNB()
NB3.fit(x_train, y_train)
score_train = NB3.score(x_train, y_train)
print('Multinomial NB training acc:',score_train)
# score_test = NB3.score(test_col, test_ans)
# print('Multinomial NB testing acc:',score_test)



if sys.argv[1]=='D':
	o=open(sys.argv[4],'a')
	for k in range(len(a)):
		o.write(str(dt_pred[k]))
		o.write("\n")
	o.close()
elif sys.argv[1]=='N':
	o=open(sys.argv[4],'a')
	for k in range(len(a)):
		o.write(str(multi_pred[k]))
		o.write("\n")
	o.close()
