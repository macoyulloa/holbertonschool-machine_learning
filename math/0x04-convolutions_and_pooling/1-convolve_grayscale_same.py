#!/usr/bin/env python3
"""Convolutional Neural Networks"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """performs a same convolurion on grayscale image
    Arg:
       images: np.array containing grayscale img (m, h, w)
       kernel: filter for the convolution (kh, kw)
    Return: convolved images np.array
    """
    m = images.shape[0]
    img_h = images.shape[1]
    img_w = images.shape[2]
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]

    p = int((k_h - 1) / 2)
    out_h = img_h - k_h + 1 + (2*p)
    out_w = img_w - k_w + 1 + (2*p)
    output_conv = np.zeros((m, out_h, out_w))

    m_img = np.arange(0, m)
    images = np.pad(images, [(0, 0), (p, p), (p, p)],
                    'constant', constant_values=0)

    for i in range(out_h):
        for j in range(out_w):
            output_conv[m_img, i, j] = np.sum(np.multiply(
                images[m_img, i:k_h+i, j:k_w+j], kernel), axis=(1, 2))
    return output_conv
