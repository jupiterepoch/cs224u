import os
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from operator import add

SUPPORT = 100

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

# create the Spark Session
spark = SparkSession.builder.getOrCreate()

# create the Spark Context
sc = spark.sparkContext

def parse_line(line):
    items = set(item for item in line.split(' ') if len(item) == 8)
    return list(items)

def get_pair(line, freq_items):
    items = list(set(item for item in line.split(' ') if item in freq_items))
    pairs = set()
    for i in range(len(items) - 1):
        for j in range(i + 1, len(items)):
            pair = tuple(sorted([items[i], items[j]]))
            pairs.add(pair)
    return list(pairs)

def get_triple(line, freq_items, freq_pairs):
    items = list(set(item for item in line.split(' ') if item in freq_items))
    triples = set()
    for i in range(len(items) - 2):
        for j in range(i + 1, len(items) - 1):
            pair = tuple(sorted([items[i], items[j]]))
            if pair in freq_pairs:
                for k in range(j + 1, len(items)):
                    triple = tuple(sorted([items[i], items[j], items[k]]))
                    triples.add(triple)
    return list(triples)

path = 'q2/data/browsing.txt'
# path = 'q2/data/small.txt'
txt = sc.textFile(path)

# first pass
candidates = txt.flatMap(parse_line).map(lambda item: (item, 1)).reduceByKey(add).filter(lambda item: item[1] >= SUPPORT).collectAsMap()
freq_items = set(candidates.keys())
print(len(candidates))

# second pass
pairs = txt.flatMap(lambda line: get_pair(line, freq_items)).map(lambda item: (item, 1)).reduceByKey(add).filter(lambda item: item[1] >= SUPPORT).collectAsMap()
freq_pairs = set(pairs.keys())
# print(pairs)

# get rules
confidence = {}
for (X, Y), count in pairs.items():
    # rule: X -> Y
    confidence[X+' -> '+Y] = count / candidates[X]
    # rule: Y -> X
    confidence[Y+' -> '+X] = count / candidates[Y]
top5 = sorted(confidence.items(), key=lambda x: (-x[1], x[0]))[:5]
for rule in top5:
    print(rule[0], '\t', rule[1])

# third pass
triples = txt.flatMap(lambda line: get_triple(line, freq_items, freq_pairs)).map(lambda item: (item, 1)).reduceByKey(add).filter(lambda item: item[1] >= SUPPORT).collectAsMap()
# print(triples)

# get rules
confidence = {}
for (X, Y, Z), count in triples.items():
    for permutes in [(X, Y, Z), (X, Z, Y), (Y, Z, X)]:
        pair = tuple(sorted([permutes[0], permutes[1]]))
        confidence['('+pair[0]+' + '+pair[1]+') -> '+permutes[2]] = count / pairs[pair]
    
top5 = sorted(confidence.items(), key=lambda x: (-x[1], x[0]))[:5]
for rule in top5:
    print(rule[0], '\t', rule[1])