#!/usr/bin/env python3
"""Convolutional Neural Networks"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """back prop convolutional 3D image, RGB image - color
    Arg:
       dA: containing the partial derivatives (m, h_new, w_new, c_new)
       A_prev: contains the output of prev layer (m, h_prev, w_prev, c)
       kernel.shape: filter dimensions tupple (kh, kw)
       stride: tuple (sh, sw)
       mode: max or avg
    Returns: parcial dev prev layer (dA_prev)
    """
    k_h, k_w = kernel_shape
    _, h_new, w_new, c_new = dA.shape
    m, h_x, w_x, c_prev = A_prev.shape
    s_h, s_w = stride

    dx = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    if mode == 'max':
                        dx[i,
                           h*(stride[0]):(h*(stride[0]))+k_h,
                           w*(stride[1]):(w*(stride[1]))+k_w,
                           :] = dA[i, h, w, f]
                    if mode == 'avg':
                        dx[i,
                           h*(stride[0]):(h*(stride[0]))+k_h,
                           w*(stride[1]):(w*(stride[1]))+k_w,
                           :] = dA[i, h, w, f]

    return dx
