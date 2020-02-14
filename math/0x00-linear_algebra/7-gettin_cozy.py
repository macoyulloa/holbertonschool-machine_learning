#!/usr/bin/env python3
""" concatenates matrix """


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenates matrix """
    if axis == 0:
        result = mat1.append(mat2[0])
        return result
    elif axis == 1:
        result = 1
        return result
    else:
        return None
