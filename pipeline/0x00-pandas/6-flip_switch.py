#!/usr/bin/env python3
""" the rows and columns are transposed and the
data is sorted in reverse chronological """

import pandas as pd
from_file = __import__('2-from_file').from_file


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df = df.T
df.sort_values('Timestamp', axis=1, inplace=True, ascending=False)

print(df.tail(8))
