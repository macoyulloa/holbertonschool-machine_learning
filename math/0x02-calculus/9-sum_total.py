#!/usr/bin/env python3
" Serie of a secuence "


def summation_i_squared(n):
    " The sum of all elem of a secuence "
    if n <= 0:
        return None
    else:
        x = sum((map(lambda x: x**2, range(1, n+1))))
        return x
