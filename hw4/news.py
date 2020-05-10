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
# %matplotlib inline

'''
train=pd.read_csv("news_train.csv",header=0)

train_col=[]
train_ans=[]
for i in range(len(train)):
    train_col.append(list(train.iloc[i,:-1]))
    train_ans.append(train.iloc[i,-1])

'''
train_col=[]
train_ans=[]
with open(sys.argv[2], newline='') as csvfile:
  rows = csv.reader(csvfile)
  for row in rows:
    row = [float(i) for i in row]
    train_col.append(row[:-1])
    train_ans.append(row[-1])


test_col=[]
# test_ans=[]
with open(sys.argv[3], newline='') as csvfile:
  rows = csv.reader(csvfile)
  for row in rows:
    row = [float(i) for i in row]
    test_col.append(row[:23909])
    # test_ans.append(row[-1])

print("parsing over\n")


x_train, x_test, y_train, y_test = model_selection.train_test_split(
    train_col, train_ans, test_size = 0.0, shuffle=True)

#decision tree

dt = tree.DecisionTreeClassifier(criterion='gini',splitter='best',
                                 max_depth=24,min_samples_split=0.09,
                                 max_features=None,max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,random_state=1)
dt.fit(x_train, y_train)
score_train = dt.score(x_train, y_train)
print('training acc:',score_train)
dt_pred=dt.predict(test_col)

# score_test = dt.score(test_col, test_ans)
# print('testing acc:',score_test)    


#naive bayes
par_smooth=0.003
# for i in range(10):

NB1 = naive_bayes.GaussianNB(var_smoothing=par_smooth)
NB1.fit(x_train, y_train)
score_train = NB1.score(x_train, y_train)
print('Gaussian NB training acc:',score_train)
# score_test = NB1.score(test_col, test_ans)
# print('Gaussian NB testing acc:',score_test)
	# par_smooth=par_smooth+0.001

NB2 = naive_bayes.BernoulliNB(alpha=0.1,fit_prior=False)
NB2.fit(x_train, y_train)
score_train = NB2.score(x_train, y_train)
print('Bernoulli NB training acc:',score_train)
# score_test = NB2.score(test_col, test_ans)
# print('Bernoulli NB testing acc:',score_test)

NB3 = naive_bayes.MultinomialNB(alpha=0.1,fit_prior=False)
NB3.fit(x_train, y_train)
score_train = NB3.score(x_train, y_train)
print('Multinomial NB training acc:',score_train)
# score_test = NB3.score(test_col, test_ans)
# print('Multinomial NB testing acc:',score_test)
multi_pred=NB3.predict(test_col)


if sys.argv[1]=='D':
	o=open(sys.argv[4],'a')
	for k in range(len(test_col)):
		o.write(str(dt_pred[k]))
		o.write("\n")
	o.close()
elif sys.argv[1]=='N':
	o=open(sys.argv[4],'a')
	for k in range(len(test_col)):
		o.write(str(multi_pred[k]))
		o.write("\n")
	o.close()