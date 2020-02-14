#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

xvals = np.arange(0, 110, 10)
plt.hist(student_grades, facecolor='b', edgecolor='black', bins=xvals)
plt.xticks(np.arange(0, 110, 10))
plt.xlim(0, 100)
plt.yticks(np.arange(0,35,5))
plt.xlabel('Grandes')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.show()
