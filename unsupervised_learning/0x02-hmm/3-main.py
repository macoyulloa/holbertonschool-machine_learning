#!/usr/bin/env python3
import csv
import numpy as np
forward = __import__('3-forward').forward

if __name__ == '__main__':
    csv_list = []
    with open('data_python.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            csv_list.append(ord(row[1])-48)
    Ob = np.array(csv_list)
    # Transition Probabilities
    Tr = np.array(((0.54, 0.46), (0.49, 0.51)))

    # Emission Probabilities
    Em = np.array(((0.16, 0.26, 0.58), (0.25, 0.28, 0.47)))

    # Equal Probabilities for the initial distribution
    Initial = np.array((0.5, 0.5)).reshape(-1, 1)

    P, F = forward(Ob, Em, Tr, Initial)
    print(P)
    print(F)
