#!/usr/bin/env python3

import numpy as np
pool_backward = __import__('3-pool_backward').pool_backward

np.random.seed(6)
m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()
c = np.random.randint(2, 5)
fh, fw = (np.random.randint(2, 5, 2)).tolist()
sh, sw = (np.random.randint(2, 5, 2)).tolist()

X = np.random.uniform(0, 1, (m, h, w, c))
dA = np.random.uniform(0, 1, (m, (h - fh) // sh + 1, (w - fw) // sw + 1, c))
Y = pool_backward(dA, X, (fh, fw), stride=(sh, sw), mode='max')
print(Y)
print(Y.shape)
