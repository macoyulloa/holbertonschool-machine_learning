#!/usr/bin/env python3

import numpy as np
conv_backward = __import__('2-conv_backward').conv_backward

np.random.seed(5)
m = np.random.randint(100, 200)
h, w = np.random.randint(20, 50, 2).tolist()
cin = np.random.randint(2, 5)
cout = np.random.randint(5, 10)
fh, fw = (np.random.randint(2, 5, 2)).tolist()
sh, sw = (np.random.randint(2, 4, 2)).tolist()

X = np.random.uniform(0, 1, (m, h, w, cin))
W = np.random.uniform(0, 1, (fh, fw, cin, cout))
b = np.random.uniform(0, 1, (1, 1, 1, cout))
dZ = np.random.uniform(0, 1, (m, h, w, cout))
dA, dW, db = conv_backward(dZ, X, W, b, padding="same", stride=(sh, sw))
np.set_printoptions(threshold=np.inf)
print(dA)
print(dA.shape)
print(dW)
print(dW.shape)
print(db)
print(db.shape)
