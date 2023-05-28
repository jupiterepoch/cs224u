import numpy as np
from matplotlib import pyplot as plt

sigma = np.exp(-5)
epsilon = np.exp(1) * 1e-4

# suffix = '_tiny'
suffix = ''

freqs = {}
with open(f'data/counts{suffix}.txt', 'r') as f:
    for line in f:
        x, count = line.strip().split('\t')
        freqs[int(x)] = int(count)

def hash_fun(a, b, x, p=123457, n_buckets=10000):
    y = x % p
    hash_val = (a*y + b) % p
    return hash_val % n_buckets

hash_params = []
counts = []

with open('data/hash_params.txt', 'r') as f:
    for line in f:
        a, b = line.strip().split('\t')
        a, b = int(a), int(b)
        hash_params.append((a, b))
        counts.append([0] * 10000)
        
def get_signature(x):
    sig = []
    for a, b in hash_params:
        sig.append(hash_fun(a, b, x))
    return sig

t = 0
with open(f'data/words_stream{suffix}.txt', 'r') as f:
    for line in f:
        t += 1
        x = int(line.strip())
        sig = get_signature(x)
        for i, h in enumerate(sig):
            counts[i][h] += 1

xs = []
ys = []
for x, count in freqs.items():
    sig = get_signature(x)
    estimate = min([counts[i][h] for i, h in enumerate(sig)])
    assert estimate >= count

    xs.append(count / t)
    ys.append((estimate - count) / count)

plt.scatter(xs, ys)
plt.xlabel('exact word frequency')
plt.ylabel('relative error')
plt.title('Relative error vs. exact word frequency')
plt.xscale('log')
plt.yscale('log')


xticks = [10**i for i in np.arange(np.floor(np.log10(min(xs))), np.ceil(np.log10(max(xs))) + 1)]
yticks = [10**i for i in np.arange(np.floor(np.log10(min(ys))), np.ceil(np.log10(max(ys))) + 1)]
plt.yticks(xticks)
plt.yticks(yticks)
plt.tight_layout()
plt.savefig(f'q4{suffix}.png')
