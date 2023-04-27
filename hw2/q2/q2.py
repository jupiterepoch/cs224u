import os
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from operator import add
import numpy as np

from matplotlib import pyplot as plt

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext


MAX_ITER = 20
NUM_CLUSTERS = 10


def euclidean_cost(p1, p2):
    return np.sum((p1 - p2) ** 2)


def manhattan_cost(p1, p2):
    return np.sum(np.abs(p1 - p2))


def load_centroids(file):
    centroids = []
    with open(file, 'r') as f:
        for line in f:
            centroid = parse_line(line)
            centroids.append(centroid)
    centroids = np.array(centroids)
    return centroids


def assign_cluster(point, centroids, cost_func):
    costs = [cost_func(point, c) for c in centroids]
    idx = np.argmin(costs)
    cost = costs[idx]
    return idx, cost


def parse_line(line):
    point = np.fromstring(line.rstrip(), dtype=np.float32, sep=' ')
    return point


if __name__ == "__main__":

    # read the input file
    data_path = "data/data.txt"
    points = sc.textFile(data_path).map(parse_line).cache()

    for cost_name, cost_func in zip(['euclidean', 'manhattan'], [euclidean_cost, manhattan_cost]):
    
        plt.figure()

        for init in ["c1", "c2"]:

            # initialize centroids
            centriods = load_centroids(f"data/{init}.txt")
            costs = []
            iterations = np.arange(MAX_ITER)
            for iteration in iterations:
                # Assign each point p to the cluster with the closest centroid
                # idx: (cost, point, 1)     1 is for counting
                assignments = points.map(lambda p: (assign_cluster(p, centriods, cost_func), p))\
                                    .map(lambda it: (it[0][0], (it[0][1], it[1], 1) ))
                
                # Recompute the centroid of each cluster as the mean of all the data points assigned to it
                new_clusters = assignments.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]))\
                                        .mapValues(lambda it: (it[0], it[1] / it[2])).collectAsMap()

                total_cost = 0
                assert len(new_clusters) == NUM_CLUSTERS
                for cluster, (cost, centroid) in new_clusters.items():
                    centriods[cluster] = centroid
                    total_cost += cost
                costs.append(total_cost)

            plt.plot(iterations, costs, label=init)
        
        plt.legend()
        plt.title(f"Cost vs Iteration using {cost_name} distance")
        plt.xticks(iterations)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.tight_layout()
        plt.savefig(f"q2_{cost_name}.png")
        plt.close()
