from pyspark import SparkContext, SparkConf
import numpy as np
import time


# ============
# Replace the code below with your code

def computeXY(rdd):
    '''
    Compute the features times targets term
    needed by OLS. Return as a numpy array
    of size (41,)
    '''

    return rdd.map(lambda r: np.multiply(add_bias(r), r[-1])).reduce(lambda x, y: x + y)


def computeXX(rdd):
    '''
    Compute the outer product term
    needed by OLS. Return as a numpy array
    of size (41,41)
    '''
    return rdd.map(lambda r: np.outer(add_bias(r), add_bias(r))).reduce(lambda x, y: x + y)


def computeWeights(rdd):
    '''
    Compute the linear regression weights.
    Return as a numpy array of shape (41,)
    '''
    return np.matmul(np.linalg.inv(computeXX(rdd)), computeXY(rdd))


def computePredictions(w, rdd):
    '''
    Compute predictions given by the input weight vector
    w. Return an RDD with one prediction per row.
    '''
    return rdd.map(lambda r: np.dot(w, add_bias(r)))


def computeError(w, rdd):
    '''
    Compute the MAE of the predictions.
    Return as a float.
    '''
    count = rdd.map(lambda r: 1).reduce(lambda x, y: x + y)
    rdd = rdd.map(lambda r: (r[-1], np.dot(w, add_bias(r))))
    return rdd.map(lambda r: abs(r[0] - r[1])).reduce(lambda x, y: x + y) / count


def add_bias(array):
    return np.append(array[:-1], 1)


# ============

# Convert rdd rows to numpy arrays
def numpyify(rdd):
    return rdd.map(lambda x: np.array(map(lambda y: float(y), x.split(","))))


# sc     = spark.sparkContext
master = "yarn"
times = []

# Flush yarn defaul context
sc = SparkContext(master, "aws_run_me")
sc.stop()

for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    conf = SparkConf().set("spark.executor.instances", i).set("spark.executor.cores", 1).set("spark.executor.memory",
                                                                                             "2G").set(
        "spark.dynamicAllocation.enabled", "false")
    sc = SparkContext(master, "aws_run_me", conf=conf)
    sc.setLogLevel("ERROR")

    start = time.time()

    rdd_test = numpyify(sc.textFile("s3://589hw03/test_data_ec2.csv")).cache()
    rdd_train = numpyify(sc.textFile("s3://589hw03/train_data_ec2.csv")).cache()

    w = computeWeights(rdd_train)
    rdd_train.persist()
    err = computeError(w, rdd_test)

    this_time = time.time() - start
    print "\n\n\nCores %d: MAE: %.4f Time: %.2f" % (i, err, this_time)
    times.append([i, this_time])

    sc.stop()

print "\n\n\n\n\n"
print times

# Cores
# 1: MAE: 0.7648
# Time: 287.59
#
# Cores
# 2: MAE: 0.7648
# Time: 150.31
#
# Cores
# 3: MAE: 0.7648
# Time: 102.05
#
# Cores
# 4: MAE: 0.7648
# Time: 80.09
#
# Cores
# 5: MAE: 0.7648
# Time: 90.17
#
# Cores
# 6: MAE: 0.7648
# Time: 92.31
#
# Cores
# 7: MAE: 0.7648
# Time: 103.10
#
# Cores
# 8: MAE: 0.7648
# Time: 100.63


#[[1, 287.59411001205444], [2, 150.31080603599548], [3, 102.04712796211243], [4, 80.09337592124939], [5, 90.166424036026], [6, 92.31054401397705], [7, 103.09541082382202], [8, 100.63411903381348]]