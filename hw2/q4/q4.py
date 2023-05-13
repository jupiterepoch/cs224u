import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

shows = []
with open("data/shows.txt", 'r') as f:
    for line in f:
        shows.append(line.rstrip())

R = []
with open("data/user-shows.txt", 'r') as f:
    for line in f:
        R.append(np.fromstring(line, dtype=int, sep=' '))
R = np.array(R)

P = np.diag(R.sum(axis=1))
Q = np.diag(R.sum(axis=0))

P_normed = np.diag(R.sum(axis=1) ** -0.5)
Q_normed = np.diag(R.sum(axis=0) ** -0.5)

Gamma_U = P_normed @ R @ R.T @ P_normed @ R # (n, m)
print("UU collaborative filtering")
u_rec = np.argsort(-Gamma_U[499, :100])[:5]
for i in u_rec:
    print(shows[i], Gamma_U[499][i])

Gamma_I = R @ Q_normed @ R.T @ R @ Q_normed # (n, m)
print("II collaborative filtering")
i_rec = np.argsort(-Gamma_I[499, :100])[:5]
for i in i_rec:
    print(shows[i], Gamma_I[499][i])
