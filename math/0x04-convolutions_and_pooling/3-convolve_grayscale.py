#!/usr/bin/env python3
"""Convolutional Neural Networks"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """performs a same convolurion on grayscale image
    Arg:
       images: np.array containing grayscale img (m, h, w)
       kernel: filter for the convolution (kh, kw)
       padding: tuple of (ph, pw), ‘same’, or ‘valid’
       stride: tuple (sh, sw)
    Return: padded convolved images np.array
    """
    m = images.shape[0]
    img_h = images.shape[1]
    img_w = images.shape[2]
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]
    s_h, s_w = stride

    if padding == 'valid':
        p_h = 0
        p_w = 0

    if padding == 'same':
        p_h = np.ceil(((s_h*img_h) - s_h + k_h - img_h) / 2)
        p_h = int(p_h)
        p_w = np.ceil(((s_w*img_w) - s_w + k_w - img_w) / 2)
        p_w = int(p_w)

    if type(padding) == tuple:
        p_h = padding[0]
        p_w = padding[1]

    images = np.pad(images, [(0, 0), (p_h, p_h), (p_w, p_w)],
                    mode='constant', constant_values=0)

    out_h = int(((img_h - k_h + (2*p_h)) / (stride[0])) + 1)
    out_w = int(((img_w - k_w + (2*p_w)) / (stride[1])) + 1)
    output_conv = np.zeros((m, out_h, out_w))
    m_img = np.arange(0, m)

    for i in range(out_h):
        for j in range(out_w):
            output_conv[m_img, i, j] = np.sum(np.multiply(
                images[m_img,
                       i*(stride[0]):k_h+(i*(stride[0])),
                       j*(stride[1]):k_w+(j*(stride[1]))],
                kernel), axis=(1, 2))
    return output_conv
