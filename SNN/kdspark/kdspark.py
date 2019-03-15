import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.cluster import DBSCAN
from scipy import spatial
from pyspark import RDD
from pyspark import SparkConf, SparkContext
import csv,io
conf = (SparkConf()
         .setMaster("local")
         .setAppName("My app")
         .set("spark.executor.cores", 4)
         .set("spark.driver.memory","7g"))

sc = SparkContext(conf = conf)
data=sc.textFile('/home/akash/Desktop/set/April.csv')
header=data.first()
data=data.filter(lambda x: x!=header)#remove header contents
data = data.map(lambda x: x.split(","))# split using appropriate delimeter
data=data.map(lambda x:(float(x[0]),float(x[1])))
data=data.filter(lambda (x,y):(int(x)<>0 and int(y)<>0) )

#latitude=data.map(lambda x: float(x[0])).cache()#extract field1
#longitude=data.map(lambda x:float(x[1])).cache()#extract field2
latitude=data.map(lambda x: np.array(x[0]).astype(float))#extract field1
longitude=data.map(lambda x: np.array(x[1]).astype(float))#extract field1


coord1=latitude.zip(longitude)# Zip latitude with longitude to coord(Spatial Information)
#print type(coord1)
coord=coord1.zipWithIndex()# Index the coordinate data--> 'coord' will be in  format(coordinate,index)

coordData = coord.map (lambda (k,v) :(v,k) )# make the index as key and coordinate data as value

count=coordData.count() # Count the number of points: count is global
countglobal=sc.broadcast(count)
ab=coord1.collect()
#print(ab[:5])
#print type(ab[0][1])
#print(ab[0][1])
tre=spatial.cKDTree(ab)
b1=coord1.take(2)
aa=sc.broadcast(tre)
graph=coordData.map(lambda (k,v):(k,tre.query(v,29)))
gra=graph.map(lambda (k,v): (k,v[1][1:]))
cra=gra.cartesian(gra)#gra=graph.map(lambda (k,v): k)
def intersectionCount(k1,k2,v1,v2):
    countNum=0
    
    if (k1 in v2 and k2 in v1) and k1!=k2:
        for i in v1:
            if i in v2:
                countNum=countNum+1
       
    if (countNum>=17):
        countNum=1
    else:
         countNum=0
    return countNum
def populateKeyValue(k1,v1,k2,v2):
    return((k1,k2),(v1,v2))
##sabi=cra.collect()
arrang=cra.map(lambda ((k1,v1),(k2,v2)):populateKeyValue(k1,v1,k2,v2))
snnGraph=arrang.map(lambda ((k1,k2),(v1,v2)):((k1,k2),intersectionCount(k1,k2,v1,v2))).cache()
bb=snnGraph.collectAsMap()
cc=sc.broadcast(bb)
#print bb
snnGraph1=snnGraph.filter(lambda x:x[1]!=0).map(lambda (((k1,k2),v)):(k1,1)) 

#i=snnGraph1.collect()
#sn=snnGraph1.filter(lambda x:x[0]==998)
y = snnGraph1.reduceByKey(lambda accum, n: accum + n)
corePointList=y.filter(lambda x:x[1]>=17).map(lambda x:x[0])
cor=corePointList.collect()
coreornot=[None for x in range(count)]
for x in cor:
    coreornot[x]=1

#print(cor)
#sharedN=snnGraph.filter(lambda ((x,y),v):x==10).map(lambda((x,y),v):v).collect()
#print sharedN
#asdf
#g=z.count()
def findCoreNeighbors(p,corePts,eps):
    coreNeighbors=[]
    p2=None
    for i in range(0,len(corePts)):
        p2=corePts[i]
        if(p==p2):
            continue
        #if two core points share more than eps neighbors make the core point core nearest neighbor of other
        if(p!=p2 and cc.value[(p,p2)]==1):
            coreNeighbors.append(p2)
    return coreNeighbors
def expandCluster(labels,neighborCore,corePts,C,eps,visited):
    while len(neighborCore)>0:
            p=neighborCore.pop(0)
            if p in visited:
                continue
            labels[p]=C
            visited.append(p)
            
            #sharedN=snnGraph.filter(lambda ((x,y),v):x==p).map(lambda((x,y),v):v).collect()        
            neighCore=findCoreNeighbors(p,corePts,eps)
            neighborCore.extend(neighCore)
    return labels
visited=[]#list to store points visited
labels=[0 for i in range(count)]
neighborCore=[]#neighborss of core points
#corePointBroadcast=sc.broadcast(corePointList)
c=0
for i in range(0,len(cor)):
    p=cor[i]
    if p in visited:
        continue
    visited.append(p)
    c=c+1
    labels[p]=c
    #sharedN=snnGraph.filter(lambda ((x,y),v):x==p).map(lambda((x,y),v):v).collect()
    neighborCore=findCoreNeighbors(p, cor, 17)   
    labels=  expandCluster(labels, neighborCore, cor, c,17, visited)
#print(c)
for i in range(count):
    notNoise=False
    maxSim=sys.maxint
    bestCore=-1
    sim=None
    #sharedNN=snnGraph.filter(lambda (x,y):(x==i)).map(lambda(x,y):y).collect()[0]
    #coreornot=coreOrNot.filter(lambda (k,v):(k==i)).map(lambda (k,v):v).collect()[0]
    if(coreornot[i]==1):#core Point
        continue
    for j in range(len(cor)):
        p=cor[j]
        #snnGraph contains count of shared neighbors between points
        # sim gives the similarity  between core point and the other point.
        
        sim=cc.value[(i,p)]
        # if sim is greater than eps--> the point is not a noise
        if(sim==1):
            notNoise=True
         # if sim is less than eps--> the point is  a noise point assign cluster index 0 to it
        else:
            labels[i]=0
            break
        #Here we attempt to see to which core point does the non-core point has maximum similarity
        if(sim>maxSim):
            maxSim=sim
            bestCore=p
        #End of inner for loop
    #for each non-core point assign the index of core point with which the point has maximum similarity
    if(notNoise):
        labels[i]=labels[bestCore]
    
