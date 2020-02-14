#!/usr/bin/env python3
" integral of a polynomium "


def poly_integral(poly, C=0):
    " function to get an integral of a poly "
    if type(poly) is not list or type(C) is not int or len(poly) == 0:
        return None
    if poly == [0]:
        return [C]
    integer = []
    integer.append(C)
    for i in range(len(poly)):
        x = poly[i]/(i + 1)
        integer.append(int(x) if x.is_integer() else x)
    return integer
