from pyspark import SparkContext, SparkConf
import numpy as np
import time

#============
#Replace the code below with your code
def computeWeights(rdd):
   return np.random.randn(41)   
   
def computeError(w,rdd_test):
   return 0     

#============

#Convert rdd rows to numpy arrays
def numpyify(rdd):
    return rdd.map(lambda x: np.array(map(lambda y: float(y),x.split(","))))

#sc     = spark.sparkContext
master = "yarn"
times=[]


#Flush yarn defaul context
sc = SparkContext(master, "aws_run_me")
sc.stop()

for i in [1,2,3,4,5,6,7,8]:

    conf = SparkConf().set("spark.executor.instances",i).set("spark.executor.cores",1).set("spark.executor.memory","2G").set("spark.dynamicAllocation.enabled","false")
    sc = SparkContext(master, "aws_run_me", conf=conf)
    sc.setLogLevel("ERROR")

    start=time.time()

    rdd_test = numpyify(sc.textFile("s3://589hw03/test_data_ec2.csv"))        
    rdd_train = numpyify(sc.textFile("s3://589hw03/train_data_ec2.csv"))

    w = computeWeights(rdd_train)
    err = computeError(w,rdd_test)
    
    this_time =  time.time()-start
    print "\n\n\nCores %d: MAE: %.4f Time: %.2f"%(i, err,this_time)
    times.append([i,this_time])

    sc.stop()

print "\n\n\n\n\n"
print times
