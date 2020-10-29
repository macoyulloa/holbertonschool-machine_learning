#!/usr/bin/env python3
" creates a pd.DataFrame from a np.ndarray "
import pandas as pd


def from_numpy(array):
    """ creates a pd.DataFrame from a np.ndarray
    """
    col = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    return (pd.DataFrame(array, columns=col[:len(array[1])]))
