from sklearn.datasets import load_iris
from scipy import spatial
import numpy as np
iris=load_iris()
X=iris.data[:,(2,3)]
X=np.array(X)
tree=spatial.KDTree(X)
k=14#Nearest Neighbor Size
minPoints=5#defines core point
eps=4 #defines noise
#print(tree.data)
ss=tree.query(X[149],14,0.3)
print(ss[1])
def compare(a,b):
	cnt=0	
	a1=tree.query(X[a],14)
	b1=tree.query(X[b],14)
	for x in a1[1]:
		for y in b1[1]:
			if x==y:
				cnt=cnt+1
	return cnt			
b=[]
for i in range(0,149):
	c=compare(149,i)	
	b.append(c)
#print(compare(148,149))
print(b)
#(x, k=1, eps=0, p=2, distance_upper_bound=inf)[source]
