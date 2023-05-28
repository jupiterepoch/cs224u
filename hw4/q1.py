import numpy as np
from numpy.linalg import svd

np.random.seed(42)
x = np.random.randint(0, 100, size=(15, 10))

def embed(x, r=9):

    u, s, vh = svd(x, full_matrices=False)
    return x @ vh.T[:, :r]


row_emb = embed(x)
col_emb = embed(x.T)

for i, emb in enumerate(row_emb):
    print('doc', i, '['+' '.join(map(lambda x: f'{x:.2f}', emb))+']')

print('-'*50)

for i, emb in enumerate(col_emb):
    print('word', i, '['+' '.join(map(lambda x: f'{x:.2f}', emb))+']')
