#!/usr/bin/env python3
" Matrix multiplication "


def mat_mul(mat1, mat2):
    " Multiplication "
    final_mat = []
    if len(mat1[0]) == len(mat2):
        for i in range(len(mat1)):
            mul = []
            for j in range(len(mat2[0])):
                result = 0
                for ii in range(len(mat1[0])):
                    result += mat1[i][ii] * mat2[ii][j]
                mul.append(result)
            final_mat.append(mul)
        return final_mat
    return None
