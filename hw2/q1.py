import numpy as np
from scipy.linalg import svd, eigh

M = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
])

u, sigma, v = svd(M, full_matrices=False)

print("U", u), print()
print("Sigma", sigma), print()
print("V^T", v), print()

Evals, Evecs = eigh(M.T @ M)
order = np.argsort(Evals)[::-1]
Evals = Evals[order]
Evecs = Evecs[:, order]

print("Eigenvalues", Evals), print()
print("Eigenvectors", Evecs), print()