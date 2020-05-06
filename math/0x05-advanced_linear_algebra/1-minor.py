#!/usr/bin/env python3
""" calculate the minor of a matrix"""


def get_minor(matrix, i, j):
    """ quit the row, i, and the column, j, per position

    Return: matrix without i and j
    """
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]


def get_matrix_determinant(minor):
    """ get the detrminant value of a matrix 2x2 dimensions

    Return: the determinant value
    """
    # base case for 2x2 matrix
    if len(minor) == 2:
        return (minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0])


def minor(matrix):
    """ Calculates the minor of the matrix
    Arg:
       - matrix: list of lists whose the minor should be calculated

    Returns:
       the minor of a matrix
    """
    shape_col = [len(row) for row in matrix]
    if (isinstance(matrix, list)) and len(matrix) is not 0:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError("matrix must be a list of lists")
    else:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(len(matrix) == col for col in shape_col):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    if len(matrix) == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    minors = []
    for row in range(len(matrix)):
        minor_row = []
        for col in range(len(matrix)):
            minor = get_minor(matrix, row, col)
            minor_row.append(get_matrix_determinant(minor))
        minors.append(minor_row)
    return minors
