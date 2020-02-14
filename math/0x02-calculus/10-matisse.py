#!/usr/bin/env python3
" derivate of a polynomial "


def poly_derivative(poly):
    " function that return a list of polynomial derivate "
    if type(poly) is not list or len(poly) is 0:
        return None
    res = []
    for x in range(len(poly)):
        res.append(poly[x]*x)
    if sum(res) == 0:
        return '[0]'
    return res[1:]
