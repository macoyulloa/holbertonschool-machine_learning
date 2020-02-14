#!/usr/bin/env python3
" Function transpose "


def matrix_transpose(matrix):
    " transpose of a 2D matrix "
    transpose = [[matrix[j][i] for j in range(len(matrix))]
                 for i in range(len(matrix[0]))]
    return transpose
