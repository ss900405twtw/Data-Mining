import pandas as pd
import sys
s=pd.read_csv(sys.argv[1],header=None)
ans=list(s.iloc[:,-1])


	
ss=pd.read_csv(sys.argv[2],header=None)
pred=list(ss[0])

correct=0
for i in range(len(s)):
	if ans[i]==pred[i]:
		correct+=1
print("accuracy is: ",correct/len(s))
