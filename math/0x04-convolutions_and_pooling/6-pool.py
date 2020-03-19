#!/usr/bin/env python3
"""Convolutional Neural Networks"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs a convolurion 3D image, RGB image - color
    Arg:
       images: np.array containing RGB img (m, h, w, c)
       kernel: filter for the convolution (kh, kw)
       stride: tuple (sh, sw)
       mood: type of pooling max or avg
    Return: padded convolved images RGB np.array
    """
    m, img_h, img_w, img_c = images.shape
    k_h = kernel_shape[0]
    k_w = kernel_shape[1]

    out_h = int(((img_h - k_h) / (stride[0])) + 1)
    out_w = int(((img_w - k_w) / (stride[1])) + 1)
    output_conv = np.zeros((m, out_h, out_w, img_c))
    m_img = np.arange(0, m)

    for i in range(out_h):
        for j in range(out_w):
            if mode == 'max':
                output_conv[m_img, i, j] = np.max(
                    images[m_img,
                           i*(stride[0]):k_h+(i*(stride[0])),
                           j*(stride[1]):k_w+(j*(stride[1]))], axis=(1, 2))
            if mode == 'avg':
                output_conv[m_img, i, j] = np.mean(
                    images[m_img,
                           i*(stride[0]):k_h+(i*(stride[0])),
                           j*(stride[1]):k_w+(j*(stride[1]))], axis=(1, 2))
    return output_conv
