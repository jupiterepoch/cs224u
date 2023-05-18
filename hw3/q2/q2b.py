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

def s2d(line):
    src, dst = line.split()
    return (int(src)-1, int(dst)-1)

def d2s(line):
    src, dst = line.split()
    return (int(dst)-1, int(src)-1)

def update(adj, x):
    dst, srcs = adj[0], adj[1]
    return (dst, np.sum([x[src] for src in srcs]))

data = sc.textFile("data/graph-full.txt")
# data = sc.textFile("data/graph-small.txt")

adj_list_a = data.map(d2s).groupByKey().mapValues(set).cache()

adj_list_h = data.map(s2d).groupByKey().mapValues(set).cache()

h = np.ones(n)

for i in range(40):

    a = np.zeros(n)
    a_updates = adj_list_a.map(lambda adj: update(adj, h)).collectAsMap()
    for dst, a_update in a_updates.items():
        a[dst] = a_update
    a /= np.max(a)

    h = np.zeros(n)
    h_updates = adj_list_h.map(lambda adj: update(adj, a)).collectAsMap()
    for dst, h_update in h_updates.items():
        h[dst] = h_update
    h /= np.max(h)


order_h = np.argsort(h)
print("hi 5 hubbiness:")
for i in range(5):
    print(order_h[-i-1] + 1, h[order_h[-i-1]])
print("lo 5 hubbiness:")
for i in range(5):
    print(order_h[i] + 1, h[order_h[i]])

order_a = np.argsort(a)
print("hi 5 authority:")
for i in range(5):
    print(order_a[-i-1] + 1, a[order_a[-i-1]])
print("lo 5 authority:")
for i in range(5):
    print(order_a[i] + 1, a[order_a[i]])
