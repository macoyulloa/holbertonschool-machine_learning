#!/usr/bin/env python3
""" adding array """


def add_arrays(arr1, arr2):
    """ adding array """
    if len(arr1) is not len(arr2):
        return None
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
