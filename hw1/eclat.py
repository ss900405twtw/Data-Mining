import sys
import time
# type = sys.getfilesystemencoding()

from sys import argv
import numpy as np
import time
def load_data(file):
    raw_data=file
    f = open(raw_data,'r')
    result = []
    for line in open(raw_data):
        line = f.readline()
        line=line.strip('\n').split(' ')
        line=[int(i) for i in line]
        result.append(line)
#         print(result)
#     print(result)
    return result,len(result)
def getC1(DB):
    C1=[]
    for Items in DB:
        for element in Items:
            if element not in C1:
                C1.append(element)
    C1.sort()
    C1=[frozenset([i]) for i in C1]
#     print(C1)
    return C1

def padding(DB,C1,minSup):
#     print("max:",C1)
    max_num=max([list(C1[i])[0] for i in range(len(C1))])
#     print(max_num)
    bitVec=[[False for j in range(max_num)] for i in range(len(DB))]
#     print("first")
#     prin
#     print(len(bitVec))
#     print(len(bitVec[0]))
    for k in range(len(DB)):
        for h in range(len(DB[k])):
              bitVec[k][int(DB[k][h])-1]=True
    #list transpose
    bitVec = [list(i) for i in zip(*bitVec)]
    new_bitVec=[]
    for item in bitVec:
        if item.count(0)!=len(bitVec[0]):
            new_bitVec.append(item)
            
    
    supData={}

            
    for i in range(len(C1)):
        if new_bitVec[i].count(1)>=minSup:
            supData[C1[i]]= np.array(new_bitVec[i])
    
    return supData,max_num
s=[]
y=[]
def eclat(prefix, items):
        while items:
            i,itids = items.pop()
            isupp = np.count_nonzero(itids)
            if isupp >= minsup:
#                 print( sorted(prefix+[i]), ':', isupp)
                s.append(prefix | i)
                y.append(isupp)
#                 print(sorted(key=lambda item: np.count_nonzero(item[1])))
#                 s.append(prefix+[i])
                suffix = []
                for j, ojtids in items:
                    jtids = itids & ojtids
                    if np.count_nonzero(jtids) >= minsup:
                        suffix.append((j,jtids))
                eclat(prefix | i, sorted(suffix, key=lambda item: np.count_nonzero(item[1]), reverse=True))
        return s
def OutFile(fine_list,output_file):
#     supList,supData=eclat(input_file,minsup)
#     print(supData)
    supData=fine_list
    
    S=[]
    sup_count=[]
    for key in supData:
        s=[]
        for elem in key:
            s.insert(-1,elem)
            s.sort()
        S.append(s)
        
        # print(S)
        sup_count.append(supData[key])
    o=open(output_file,'a')
    for k in range(len(sup_count)):
        for i in S[k]:
            o.write(str(i))
            o.write(" ")
        o.write("(")
        o.write(str(sup_count[k]))
        o.write(")")
        o.write("\n")
    o.close()
    return S,sup_count



str1 = sys.argv[1]

ratio = float(sys.argv[2])



file=str1
t0=time.time()
DB ,num= load_data(file)
minsup = num*ratio
C1 = getC1(DB)
bitVec,max_num=padding(DB,C1,minsup)
# minsup = 8416*ratio
q=eclat(set(), sorted(bitVec.items(), key=lambda item: np.count_nonzero(item[1]), reverse=True))

di={}
i=0
for item in s:
    di[frozenset(item)]=y[i]
    i+=1

t1=time.time()
# print("first take time: ",t1-t0," sec")
OutFile(di,sys.argv[3])