from pyspark import RDD
from pyspark import SparkConf, SparkContext
import csv,io
conf = (SparkConf()
         .setMaster("local")
         .setAppName("My app")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)
data=sc.textFile('/home/akash/Desktop/set/April.csv')
header=data.first()
data=data.filter(lambda x: x!=header).cache()#remove header contents
data = data.map(lambda x: x.split(",")).cache()# split using appropriate delimeter
data=data.map(lambda x:(float(x[0]),float(x[1])))
data=data.filter(lambda (x,y):(int(x)<>0 and int(y)<>0) )
latitude=data.map(lambda x: float(x[0])).cache()#extract field1
longitude=data.map(lambda x:float(x[1])).cache()#extract field2
coord=latitude.zip(longitude)# Zip latitude with longitude to coord(Spatial Information)
coord=coord.zipWithIndex()# Index the coordinate data--> 'coord' will be in  format(coordinate,index)
coordData = coord.map (lambda (k,v) :(v,k) )# make the index as key and coordinate data as value
global count
count=coordData.count() # Count the number of points: count is global
countglobal=sc.broadcast(count)
def eucledian_dist(latlong1, latlong2):
    lat1, lon1 = latlong1
    lat2, lon2 = latlong2
    return ((float(lat2)-float(lat1))**2+(float(lon2)-float(lon1))**2)**0.5
def computeDistanceMatrix(coordData,count):
    index=range(count)# index contains values 0 through count-1
    indices = sc.parallelize([(i,j) for i in index for j in index if i < j])# contains all the upper traingular indices. e.g. (0,1)
    joined1 = indices.join(coordData).map(lambda (i, (j, val)): (j, (i, val))).cache()#join operation of coordData on indices. e.g. (1, (0, (-73.97205, 40.75908)))
    joined2 = joined1.join(coordData).map(lambda (j, ((i, latlong1), latlong2)): ((i,j), (latlong1, latlong2)) ).cache()# form key value pair.
    #..Key= indices of two points, value=attributes of two points. e.g.((0, 4), ((-73.97205, 40.75908), (-73.99279, 40.76308)))
    distanceMatrix=joined2.mapValues(lambda (x, y): eucledian_dist(x, y)).coalesce(3).cache()#compute pairwise distance..
    return distanceMatrix
distanceMatrix=computeDistanceMatrix(coordData,count)
