#!/usr/bin/env python3
"""Convolutional Neural Networks"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """performs a convolurion 3D image, RGB image - color
    Arg:
       images: np.array containing RGB img (m, h, w, c)
       kernel: filter for the convolution (kh, kw, c, nc)
       padding: tuple of (ph, pw), ‘same’, or ‘valid’
       stride: tuple (sh, sw)
    Return: padded convolved images RGB np.array
    """
    m, img_h, img_w, img_c = images.shape
    k_h, k_w, k_c, k_nc = kernels.shape

    if padding == 'valid':
        p_h = 0
        p_w = 0

    if padding == 'same':
        p_h = int((k_h - 1) / 2)
        p_w = int((k_h - 1) / 2)

    if type(padding) == tuple:
        p_h = padding[0]
        p_w = padding[1]

    images = np.pad(images, [(0, 0), (p_h, p_h), (p_w, p_w), (0, 0)],
                    mode='constant', constant_values=0)

    out_h = int(((img_h - k_h + (2*p_h)) / (stride[0])) + 1)
    out_w = int(((img_w - k_w + (2*p_w)) / (stride[1])) + 1)

    output_conv = np.zeros((m, out_h, out_w, k_nc))
    m_img = np.arange(0, m)

    for i in range(out_h):
        for j in range(out_w):
            for f in range(k_nc):
                output_conv[m_img, i, j, f] = np.sum(np.multiply(
                    images[m_img,
                           i*(stride[0]):k_h+(i*(stride[0])),
                           j*(stride[1]):k_w+(j*(stride[1]))],
                    kernels[f]), axis=(1, 2, 3))
    return output_conv
