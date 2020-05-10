import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing
import pickle



s=pd.read_csv(sys.argv[2],header=None)
a=pd.read_csv(sys.argv[3],header=None)

#training data

#training data
s_category=s.iloc[:,[1,3,5,6,7,8,9,13]]
s_continuous=s.iloc[:,[0,4,10,11,12,14]]
train_col_cat=[]

for i in range(len(s_category)):
    train_col_cat.append(list(s_category.iloc[i,0:]))




# input(list(s_category))
# input(list(train_col_cat))
'''
for col in list(s_category):
    b, c = np.unique(s_category[col], return_inverse=True) 
    s_category[col] = c

for i in range(len(s_category)):
    train_col_cat.append(list(s_category.iloc[i,0:]))	
'''
# s_category=s_category.toarray()
# input(train_col_cat[0:3])










enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
# enc=preprocessing.LabelEncoder(handle_unknown='ignore')
enc.fit(train_col_cat)
train_col_cat = enc.transform(train_col_cat).toarray()
# input(train_col_cat[0:3])

train_col_cat=[list(train_col_cat[i]) for i in range(len(train_col_cat))]




train_col_cont=[]
train_ans=[]
for i in range(len(s_category)):
    train_col_cont.append(list(s_continuous.iloc[i,0:-1]))
    train_ans.append(s_continuous.iloc[i,-1])
train_col=[train_col_cat[i]+train_col_cont[i] for i in range(len(train_col_cont))]


#testing data

a_category=a.iloc[:,[1,3,5,6,7,8,9,13]]
a_continuous=a.iloc[:,[0,4,10,11,12]]
test_col_cat=[]

for i in range(len(a_category)):
    test_col_cat.append(list(a_category.iloc[i,0:]))

test_col_cat = enc.transform(test_col_cat).toarray()
test_col_cat=[list(test_col_cat[i]) for i in range(len(test_col_cat))]

test_col_cont=[]
# train_ans=[]
for i in range(len(a_category)):
    test_col_cont.append(list(a_continuous.iloc[i,0:]))
    # test_ans.append(s_continuous.iloc[i,-1])
test_col=[test_col_cat[i]+test_col_cont[i] for i in range(len(test_col_cont))]
# enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
# enc.fit(train_col_cat)
'''
test_col_cat = enc.transform(test_col_cat).toarray()
test_col_cat=[list(test_col_cat[i]) for i in range(len(test_col_cat))]

test_col_cont=[]
# train_ans=[]
for i in range(len(a_category)):
    test_col_cont.append(list(a_continuous.iloc[i,0:]))
    # test_ans.append(s_continuous.iloc[i,-1])
test_col=[test_col_cat[i]+test_col_cont[i] for i in range(len(test_col_cont))]
'''




o=open('income.tr','a')
for k in range(len(s)):
	o.write(str(train_ans[k]))
	o.write(" ")
	for i in range(len(train_col[0])):
		o.write(str(i+1))
		o.write(":")
		o.write(str(train_col[k][i]))
		o.write(" ")
	o.write("\n")
o.close()

o=open('income.te','a')
for k in range(len(a)):
  o.write(str(0))
  o.write(" ")
  for i in range(len(test_col[0])):
    o.write(str(i+1))
    o.write(":")
    o.write(str(test_col[k][i]))
    o.write(" ")
  o.write("\n")
o.close()
'''
o=open('income.te','a')
for k in range(len(a)):
	o.write(str(train_ans[k]))
	o.write(" ")
	for i in range(len(test_col[0])):
		o.write(str(i+1))
		o.write(":")
		o.write(str(test_col[k][i]))
		o.write(" ")
	o.write("\n")
o.close()
'''
