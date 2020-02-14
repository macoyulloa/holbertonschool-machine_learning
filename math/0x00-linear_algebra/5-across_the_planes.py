#!/usr/bin/env python3
""" adding matrix """


def add_matrices2D(mat1, mat2):
    """ Adding matrix """
    if (len(mat1) == len(mat2)) & (len(mat1[0]) == len(mat2[0])):
        result = [[i+j for i, j in zip(x, y)] for x, y in zip(mat1, mat2)]
        return result
    return None
