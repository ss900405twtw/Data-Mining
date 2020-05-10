import numpy as np
import sys
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
    return result,len(result)

def getC1(DB):
    C1=[]
    for Items in DB:
        for element in Items:
            if element not in C1:
                C1.append(element)
    C1.sort()
    C1=[frozenset([i]) for i in C1]
    return C1
def scanDB(DB,Ck,minSup):
    ItemSet={}
    for i in DB:
        for j in Ck:
            if j.issubset(i):	
                if j not in ItemSet:    
                    ItemSet[j]=1
                else:
                    ItemSet[j] += 1
                    
    supList=[]
    supData={}
    for k in ItemSet:
        if ItemSet[k] >= minSup:
            supList.insert(0,k)
            supData[k]=ItemSet[k]
    
    return supList,supData
def canGen(Lk,k):
    genCan=[]
    for i in range(len(Lk)):
        for j in range(i+1,len(Lk)):
                L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
                L1.sort(); L2.sort()
                if L1==L2: #if first k-2 elements are equal
                    genCan.append(Lk[i] | Lk[j]) #set union
    return genCan 

def Apriori(file, minSupport = 4000):

#     print("C1 lenth: ",len(C1))
    
    rawData,num=load_data(file)
    c1 = getC1(rawData)
    DB = [set(i) for i in rawData]
    # DB = list(map(set, rawData))
    minSupport=minSupport*num
    supList, supportData = scanDB(DB, c1, minSupport)
    L = [supList]
    # print("L: ",L)
#     print("supp ",supportData)
    k = 2
    while (len(L[k-2]) > 0):
        Ck = canGen(L[k-2], k)
        Lk, supK = scanDB(DB, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
#         print(L)
        k += 1
    print(len(L))
    return L, supportData
def OutFile(input_file,minsup,output_file):
    supList,supData=Apriori(input_file,minsup)
#     print(supData)
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



t0 = time.time()

OutFile(sys.argv[1],float(sys.argv[2]),sys.argv[3])
t1 = time.time()
total = t1-t0
print("total time ",total)