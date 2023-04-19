import os
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from operator import add

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

# create the Spark Session
spark = SparkSession.builder.getOrCreate()

# create the Spark Context
sc = spark.sparkContext

def get_real_friends(row):
    pairs = []
    user, friends = row.split('\t')
    if not friends:
        return pairs
    user = int(user)
    for f in friends.split(','):
        pairs.append((user, int(f)))
        # pairs.append((int(f), user))
    return pairs

def get_potential_recs(row):
    pairs = []
    _, friends = row.split('\t')
    if not friends:
        return pairs
    friends = [int(friend_id) for friend_id in friends.split(',')]
    if len(friends) < 2:
        return pairs
    for i in range(len(friends) - 1):
        for j in range(i + 1, len(friends)):
            pairs.append((friends[i], friends[j]))
            pairs.append((friends[j], friends[i]))
    return pairs

def rank_friends(friends):
    friends = sorted(friends, key=lambda x: (-x[1], x[0]))
    return [f[0] for f in friends[:10]]

path = 'q1/data/soc-LiveJournal1Adj.txt'
# path = 'q1/data/small.txt'
txt = sc.textFile(path)
real_friends = txt.flatMap(get_real_friends)
friends = txt.flatMap(get_potential_recs).subtract(real_friends)
counts = friends.map(lambda p: (p, 1)).reduceByKey(add)
user2counts = counts.map(lambda triple: (triple[0][0], (triple[0][1], triple[1]))).groupByKey().mapValues(list)
ranked_list = user2counts.map(lambda it: (it[0], rank_friends(it[1])))
query = [11, 924, 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993]
for q in query:
    print(q, '\t', ranked_list.lookup(q)[0])
