#!/usr/bin/env python3
""" From File """
import pandas as pd


def from_file(filename, delimiter):
    """ loads data from a file as a pd.DataFrame

        Returns: the loaded pd.DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
