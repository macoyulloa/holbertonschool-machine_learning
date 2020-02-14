#!/usr/bin/env python3
" Function dimension of a matrix "


def matrix_shape(matrix):
    " Dimension of a matrix "
    if not type(matrix) == list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
