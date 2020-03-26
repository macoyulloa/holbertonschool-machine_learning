#!/usr/bin/env python3
"""Convolutional Neural Networks"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """back prop convolutional 3D image, RGB image - color
    Arg:
       dZ: containing the partial derivatives (m, h_new, w_new, c_new)
       A_prev: contains the output of prev layer (m, h_prev, w_prev, c_prev)
       W: filter for the convolution (kh, kw, c_prev, c_new)
       b: biases (1, 1, 1, c_new)
       padding: string ‘same’, or ‘valid’
       stride: tuple (sh, sw)
    Returns: parcial dev prev layer (dA_prev), kernels (dW), biases (db)
    """
    k_h, k_w, c_prev, c_new = W.shape
    m, h_x, w_x, c_prev = A_prev.shape
    s_h, s_w = stride

    if padding == 'valid':
        p_h = 0
        p_w = 0

    if padding == 'same':
        p_h = int(((s_h * h_prev) - s_h + k_h - h_prev) / 2) + 1
        p_w = int(((s_w * w_prev) - s_w + k_w - w_prev) / 2) + 1

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    x_padded = np.pad(A_prev, [(0, 0), (p_h, p_h), (p_w, p_w), (0, 0)],
                      mode='constant', constant_values=0)

    x_padded_bcast = np.expand_dims(x_padded, axis=-1)
    dZ_bcast = np.expand_dims(dZ, axis=-2)

    dW = np.zeros_like(W)
    for h in range(k_h):
        for w in range(k_w):
            dW[h, w, :, :] = np.sum(dZ_bcast *
                                    x_padded_bcast[
                                        :,
                                        h*s_h:h_x-(k_h-1-(h*s_h)),
                                        w*s_w:w_x-(k_w-1-(w*s_w)),
                                        :, :],
                                    axis=(0, 1, 2))

    dx = np.zeros_like(x_padded, dtype=float)
    Z_p_h = k_h - 1
    Z_p_w = k_w - 1
    dZ_padded = np.pad(A_prev, [(0, 0), (Z_p_h, Z_p_h),
                                (Z_p_w, Z_p_w), (0, 0)],
                       mode='constant', constant_values=0)

    return dx, dW, db
