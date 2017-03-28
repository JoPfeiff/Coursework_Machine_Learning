
import numpy as np
import time

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint,  LinearRegressionModel, LassoWithSGD, RidgeRegressionWithSGD
from pyspark.sql import SQLContext
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession

from pyspark.mllib.feature import ChiSqSelector

#============
#Replace the code below with your code
    
def computeXY(rdd):    
    '''
    Compute the features times targets term
    needed by OLS. Return as a numpy array
    of size (41,)
    '''

    return rdd.map(lambda r: np.multiply(add_bias(r),r[-1])).reduce(lambda x,y: x+y)

def computeXX(rdd):
    '''
    Compute the outer product term
    needed by OLS. Return as a numpy array
    of size (41,41)
    '''    
    return rdd.map(lambda r: np.outer(add_bias(r),add_bias(r))).reduce(lambda x,y: x+y)
    
def computeWeights(rdd):  
    '''
    Compute the linear regression weights.
    Return as a numpy array of shape (41,)
    '''
    return np.matmul(np.linalg.inv(computeXX(rdd)),computeXY(rdd))
    
def computePredictions(w,rdd):  
    '''
    Compute predictions given by the input weight vector
    w. Return an RDD with one prediction per row.
    '''
    return rdd.map(lambda r: np.dot(w,add_bias(r)))
    
def computeError(w,rdd):
    '''
    Compute the MAE of the predictions.
    Return as a float.
    '''
    count = rdd.map(lambda r: 1).reduce(lambda x,y: x+y)
    rdd = rdd.map(lambda r: (r[-1], np.dot(w,add_bias(r))))
    return rdd.map(lambda r: abs(r[0]-r[1])).reduce(lambda x,y: x+y)/count

def add_bias(array):
    return np.append(array[:-1],1)



def get_denom(rdd, min):
    max = rdd.reduce(lambda x,y: np.maximum(x,y))
    return   max - min

def get_min(rdd):
    return rdd.reduce(lambda x,y: np.minimum(x,y))

def normalize(rdd, denom, min):
    return rdd.map(lambda r: np.append((r[:-1] - min[:-1]) / denom[:-1],r[-1]) )




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
output = []


####################################################
# Normalization
####################################################


for i in [4]:
    conf = SparkConf().set("spark.executor.instances", i).set("spark.executor.cores", 1) \
        .set("spark.executor.memory", "2G").set("spark.dynamicAllocation.enabled", "false")
    sc = SparkContext(master, "aws_run_me", conf=conf)
    sc.setLogLevel("ERROR")

    start = time.time()
    spark = SparkSession.builder \
        .master("local") \
        .appName("Word Count") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    rdd_test = numpyify(sc.textFile("s3://589hw03/test_data_ec2.csv")).cache()
    rdd_train = numpyify(sc.textFile("s3://589hw03/train_data_ec2.csv")).cache()

    min = get_min(rdd_train)
    denom = get_denom(rdd_train,min)

    rdd_train = normalize(rdd_train, denom, min )
    rdd_test = normalize(rdd_test, denom, min )

    w = computeWeights(rdd_train)
    err = computeError(w,rdd_test)

    this_time = time.time() - start
    print "\n\n\nCores %d: MAE: %.4f Time: %.2f " % (i, err, this_time)
    text = "Cores %d: MAE: %.4f Time: %.2f " % (i, err, this_time)
    output.append([text])
    times.append([i, this_time])
    f1 = open('output.txt', 'w+')
    f1.write(str(output))
    sc.stop()


####################################################
# Lasso
####################################################


for p in  np.arange(0.0,0.5,0.05):
    for i in [4]:

        conf = SparkConf().set("spark.executor.instances",i).set("spark.executor.cores",1)\
            .set("spark.executor.memory","2G").set("spark.dynamicAllocation.enabled","false")
        sc = SparkContext(master, "aws_run_me", conf=conf)
        sc.setLogLevel("ERROR")

        start=time.time()
        spark = SparkSession.builder \
         .master("local") \
         .appName("Word Count") \
         .config("spark.some.config.option", "some-value") \
         .getOrCreate()

        rdd_test = numpyify(sc.textFile("s3://589hw03/test_data_ec2.csv")).cache()
        rdd_train = numpyify(sc.textFile("s3://589hw03/train_data_ec2.csv")).cache()

        rdd_test = rdd_test.map(lambda r: LabeledPoint(r[-1], r[:-1])).cache()
        rdd_train = rdd_train.map(lambda r: LabeledPoint(r[-1], r[:-1])).cache()

        model = LassoWithSGD.train(rdd_train, regParam = p)

        valuesAndPreds = rdd_test.map(lambda p: (p.label, model.predict(p.features)))
        err = valuesAndPreds.map(lambda (v, p): abs(v - p)).reduce(lambda x, y: x + y) / valuesAndPreds.count()

        this_time =  time.time()-start
        print "\n\n\nCores %d: MAE: %.4f Time: %.2f p: %.2f"%(i, err,this_time,p)
        text = "Cores %d: MAE: %.4f Time: %.2f p: %.2f Lasso"%(i, err,this_time,p)
        output.append([text])
        times.append([i,this_time,p])
        # f1 = open('output.txt', 'w+')
        # f1.write(str(output))
        sc.stop()


####################################################
# Ridge
####################################################

for p in  np.arange(0.0,0.5,0.05):
    for i in [4]:

        conf = SparkConf().set("spark.executor.instances",i).set("spark.executor.cores",1)\
            .set("spark.executor.memory","2G").set("spark.dynamicAllocation.enabled","false")
        sc = SparkContext(master, "aws_run_me", conf=conf)
        sc.setLogLevel("ERROR")

        start=time.time()

        spark = SparkSession.builder \
         .master("local") \
         .appName("Word Count") \
         .config("spark.some.config.option", "some-value") \
         .getOrCreate()

        rdd_test = numpyify(sc.textFile("s3://589hw03/test_data_ec2.csv")).cache()
        rdd_train = numpyify(sc.textFile("s3://589hw03/train_data_ec2.csv")).cache()

        rdd_test = rdd_test.map(lambda r: LabeledPoint(r[-1], r[:-1])).cache()
        rdd_train = rdd_train.map(lambda r: LabeledPoint(r[-1], r[:-1])).cache()

        model = RidgeRegressionWithSGD.train(rdd_train, regParam = p)

        valuesAndPreds = rdd_test.map(lambda p: (p.label, model.predict(p.features)))
        err = valuesAndPreds.map(lambda (v, p): abs(v - p)).reduce(lambda x, y: x + y) / valuesAndPreds.count()

        this_time =  time.time()-start
        print "\n\n\nCores %d: MAE: %.4f Time: %.2f p: %.2f"%(i, err,this_time,p)
        text = "Cores %d: MAE: %.4f Time: %.2f p: %.2f Ridge"%(i, err,this_time,p)
        output.append([text])
        times.append([i,this_time,p])
        # f1 = open('output.txt', 'w+')
        # f1.write(str(output))
        sc.stop()


# f1=open('output.txt', 'w+')
# f1.write(str(output))
print times
