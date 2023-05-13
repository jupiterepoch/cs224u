import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

np.random.seed(0)

k = 20
reg = 0.1
epochs = 50
eta = 2e-2

train = "data/ratings.train.txt"

m = n = 0
with open(train, 'r') as f:
    for line in f:
        uid, mid, _ = map(lambda x: int(x), line.rstrip().split())
        n = max(n, uid)
        m = max(m, mid)
p = np.random.rand(n+1, k) * np.sqrt(5 / k)
q = np.random.rand(m+1, k) * np.sqrt(5 / k)

costs = []
for i in tqdm(range(epochs)):
    with open(train, 'r') as f:
        for line in f:
            uid, mid, rating = map(lambda x: int(x), line.rstrip().split())
            epsilon = 2 *  (rating - np.dot(p[uid], q[mid]))
            delta_p = eta * (epsilon * q[mid] - 2 * reg * p[uid])
            delta_q = eta * (epsilon * p[uid] - 2 * reg * q[mid])
            p[uid] += delta_p
            q[mid] += delta_q
    
    E = reg * (np.sum(p ** 2) + np.sum(q ** 2))
    with open(train, 'r') as f:
        for line in f:
            uid, mid, rating = map(lambda x: int(x), line.rstrip().split())
            E += (rating - np.dot(p[uid], q[mid])) ** 2

    costs.append(E)

plt.plot(np.arange(epochs), costs)
# plt.xticks(np.arange(epochs))
plt.xlabel("Iteration")
plt.ylabel("E")
plt.title("E vs Iteration")
plt.tight_layout()
plt.savefig("q3.png")
