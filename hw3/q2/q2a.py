import os
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from operator import add

import numpy as np

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

n = 1000
m = 8192
beta = 0.8

def s2d(line):
    src, dst = line.split()
    return (int(src)-1, int(dst)-1)

def d2s(line):
    src, dst = line.split()
    return (int(dst)-1, int(src)-1)

def update_contributions(adj, r, degrees):
    dst, srcs = adj[0], adj[1]
    return (dst, np.sum([r[src] / degrees[src] for src in srcs]))

data = sc.textFile("data/graph-full.txt")
# data = sc.textFile("data/graph-small.txt")

degrees = data.map(s2d).groupByKey().mapValues(set).map(lambda x: (x[0], len(x[1]))).collectAsMap()

adj_list = data.map(d2s).groupByKey().mapValues(set)# .map(lambda x: (x[0], np.unique(x[1])))

#.collectAsMap()
r = np.ones(n) / n

for i in range(40):
    contributions = adj_list.map(lambda adj: update_contributions(adj, r, degrees)).collectAsMap()
    r = (np.ones(n) - beta) / n
    for dst, contribution in contributions.items():
        r[dst] += beta * contribution


order = np.argsort(r)

print("Top 5 pages:")
for i in range(5):
    print(order[-i-1] + 1, r[order[-i-1]])

print("Bottom 5 pages:")
for i in range(5):
    print(order[i] + 1, r[order[i]])