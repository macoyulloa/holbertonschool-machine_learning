#!/usr/bin/env python3
" creates a pd.DataFrame from a np.ndarray "
import pandas as pd


d = {'First': [0.0, 0.5, 1.0, 1.5], 'Second': ['one', 'two', 'three', 'four']}
df = pd.DataFrame(data=d, index=['A', 'B', 'C', 'D'])
