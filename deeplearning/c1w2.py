# %% startup

import numpy as np
import time

a = np.random.rand(2,2)
a = np.reshape(a,-1)
print(a)

# %% Testing loop vs vectorization
a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(c)
print('Vectorized version:', str(1000 * (toc - tic)) + ' ms')

c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] + b[i]
toc = time.time()

print(c)
print('For-Loop version:', str(1000 * (toc - tic)) + ' ms')

# %% Broadcasting example

A = np.array([[56.0,0.0,4.4,68.0],
            [1.2,104.0,52.0,8.0],
            [1.8,135.0,0.99,0.9]])

print(A)

# %%
cal = A.sum(axis=0)
print(cal)
percentage = 100*A/(cal.reshape(1,4))
print(percentage)

# %%
print(a.shape)

    # %%
a.shape[0]

# %%
v = np.random.rand(3,1000)
print(v)


# %%
v[1,1:64]

# %%


# %%
