import math
import pandas as pd
import numpy as np
from utils import swap

def DetGreedy(data,p,k_max):
    rankedAttList = [] 
    rankedScoreList = []
    counts={}
    for i in range(len(p)):
        counts[i]=0
    for k in range(1,k_max+1):
        belowMin = {ai for ai,v in counts.items() if v < math.floor(k*p[ai]) }
        belowMax = {ai for ai,v in counts.items() if v >= math.floor(k*p[ai]) and v <math.ceil(k*p[ai]) }
        s={}
        if len(belowMin) != 0:
            for i in belowMin:
                s[i]=data[(i,counts[i])]
            nextAtt = max(s,key=s.get)
        else:
            for i in belowMax:
                s[i]=data[(i,counts[i])]
            nextAtt = max(s,key=s.get)
        rankedAttList.append(nextAtt)
        rankedScoreList.append(data[(nextAtt,counts[nextAtt])])
        counts[nextAtt]+=1
    return pd.DataFrame(list(zip(rankedAttList, rankedScoreList)),columns =['ai', 'score'])


def DetCons(data,p,kmax):

    rankedAttrList=[]
    rankedScoreList=[]
    counts={}
    for i in range(len(p)):
        counts[i]=0
    
    for k in range(1,kmax+1):
        belowMin = {ai for ai,v in counts.items() if v < math.floor(k*p[ai]) }
        belowMax = {ai for ai,v in counts.items() if v >= math.floor(k*p[ai]) and v <math.ceil(k*p[ai]) }
        s={}
        if len(belowMin)!=0:
            for i in belowMin:
                s[i]=data[(i,counts[i])]
            nextAtt= max(s,key=s.get)
                
        else: 
            s={}
            for i in belowMax:
                s[i]=math.ceil(i*p[i])/p[i]
                
            nextAtt= min(s,key=s.get)
                
        rankedAttrList.append(nextAtt)
        rankedScoreList.append(data[(nextAtt,counts[nextAtt])])
        counts[nextAtt]+=1
        
    return pd.DataFrame(list(zip(rankedAttrList, rankedScoreList)),columns =['ai', 'score'])

def DetRelax(data,p,kmax):

    rankedAttrList=[]
    rankedScoreList=[]
    counts={}
    for i in range(len(p)):
        counts[i]=0
    
    for k in range(1,kmax+1):
        belowMin = {ai for ai,v in counts.items() if v < math.floor(k*p[ai]) }
        belowMax = {ai for ai,v in counts.items() if v >= math.floor(k*p[ai]) and v <math.ceil(k*p[ai]) }
        s={}
        if len(belowMin)!=0:
            for i in belowMin:
                s[i]=data[(i,counts[i])]
            nextAtt= max(s,key=s.get)
                
        else: 
            ns={}
            for i in belowMax:
                ns[i]=math.ceil(math.ceil(i*p[i])/p[i])
            temp = min(ns.values()) 
            nextAttSet = [key for key in ns if ns[key] == temp] 
            for i in nextAttSet:
                s[i]=data[(i,counts[i])]
            
            nextAtt= max(s,key=s.get)
                
        rankedAttrList.append(nextAtt)
        rankedScoreList.append(data[(nextAtt,counts[nextAtt])])
        counts[nextAtt]+=1
        
    return pd.DataFrame(list(zip(rankedAttrList, rankedScoreList)),columns =['ai', 'score'])

def DetConstSort(data,p,kmax):

    rankedAttrDict={}
    rankedScoreDict={}
    maxIndicesDict={}
    
    counts={}
    minCounts={}
    tempMinCounts={}

    lastEmpty=0
    k=0
    
    
    for i in range(len(p)):
        counts[i]=0
        minCounts[i]=0
    
    while lastEmpty<kmax:
        k+=1
        for j in range(len(p)):
            tempMinCounts[j]=math.floor(k*p[j])
        
        changedMins={ai for ai,s in minCounts.items() if s<tempMinCounts[ai]}
        if len(changedMins) != 0:
            vals={}
            for ai in changedMins:
                vals[ai]=data[(ai,counts[ai])]
            ordChangedMins=np.asarray((sorted(vals.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)))[:,0].tolist()
            for ai in ordChangedMins:
                rankedAttrDict[lastEmpty]=ai
                rankedScoreDict[lastEmpty]=data[(ai,counts[ai])]
                maxIndicesDict[lastEmpty]=k
                start=lastEmpty
                
                while start>0 and maxIndicesDict[start-1]>=maxIndicesDict[start] and rankedScoreDict[start-1]<rankedScoreDict[start]:
                    swap(maxIndicesDict,start-1,start)
                    swap(rankedAttrDict,start-1,start)
                    swap(rankedScoreDict,start-1,start)
                    start-=1
                counts[ai]+=1
                lastEmpty+=1
            minCounts=tempMinCounts.copy()

    rankedAttrList=[ v for v in rankedAttrDict.values() ]
    rankedScoreList=[ v for v in rankedScoreDict.values() ]
    return pd.DataFrame(list(zip(rankedAttrList, rankedScoreList)),columns =['ai', 'score'])