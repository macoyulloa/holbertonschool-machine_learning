#!/usr/bin/env python3
""" Linear and Matricial Algebra"""


def get_determinant_multiD(matrix, total=0):
    """ get detemrinant of a multi dimensional array
    """
    columns = list(range(len(matrix)))
    # get the column indices
    if len(matrix) == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    for col in columns:
        copy_matrix = [row[:] for row in matrix]
        copy_matrix = copy_matrix[1:]
        # quit the first row, I do not need it
        rows = len(copy_matrix)

        for i in range(rows):
            # quit the column that I do not need per sub-determinant
            copy_matrix[i] = copy_matrix[i][0:col] + copy_matrix[i][col+1:]
        sign = (-1) ** (col % 2)
        # give the sign depends on the position[ij] of the matrix
        sub_determinant = get_determinant_multiD(copy_matrix)
        # do all the sub_deteminants depends on the size of te matrix
        total += sign * matrix[0][col] * sub_determinant

    return total


def determinant(matrix):
    """ Calculates the detemrinant of a matrix
    Arg:
       - matrix: list of lists whose determinant should be calculated
    Returns:
       the detemrinant of a matrix
    """
    if matrix == [[]]:
        return 1

    shape_col = [len(row) for row in matrix]
    if (isinstance(matrix, list)) and len(matrix) is not 0:
        if not all(isinstance(row, list) for row in matrix):
            raise TypeError("matrix must be a list of lists")
    else:
        raise TypeError("matrix must be a list of lists")
    if not all(len(matrix) == col for col in shape_col):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    return get_determinant_multiD(matrix)


def get_minor(matrix, i, j):
    """ quit the row, i, and the column, j, per position

    Return: matrix without i and j
    """
    minor = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
    return determinant(minor)


def inverse(matrix):
    """ Calculates the inverse of the matrix
    Arg:
       - matrix: list of lists whose the inverse should be calculated

    Returns:
       the inverse of a matrix
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

    # if matrix has just one value
    if len(matrix) == 1:
        if matrix[0][0] == 0:
            return None
        else:
            return [[(1 / matrix[0][0])]]
    # inversed if the matrix is singular, return None
    if determinant(matrix) is 0:
        return None
    # get the factor: 1 / |A|, A is the determinant
    factor = 1 / determinant(matrix)
    # inversed if the matrix is a square 2x2
    if len(matrix) == 2:
        adj = [[matrix[1][1], -matrix[0][1]], [-matrix[1][0], matrix[0][0]]]
        return [[(adj[i][j] * factor) for j in range(
            len(matrix))] for i in range(len(matrix[0]))]

    coeficient = []
    # inversed for a multidimensional matrix
    for row in range(len(matrix)):
        coeficient_row = []
        for col in range(len(matrix)):
            # getting the minor values
            minor = get_minor(matrix, row, col)
            sign = (-1) ** (col + row)
            # calculates the coeficient, multiply by the position sign
            coeficient_row.append(sign * minor)
        coeficient.append(coeficient_row)
    # transversing the coefi matrix and multiply by the factor
    inversed = [[(coeficient[j][i] * factor) for j in range(
        len(coeficient))] for i in range(len(coeficient[0]))]

    return inversed
