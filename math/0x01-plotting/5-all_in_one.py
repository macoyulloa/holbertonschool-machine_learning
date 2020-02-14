#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.figure()

" linear "
plt.subplot(321)
plt.plot(y0, 'r')
plt.yticks(np.arange(0, 1500, 500))
plt.xlim(0, 10)

" scatter "
plt.subplot(322)
plt.scatter(x1, y1, c='magenta', marker='o', s=10)
plt.title('Men\'s Height vs Weight', size='x-small')
plt.xlabel('Height (in)', size='x-small')
plt.ylabel('Weight (lbs)', size='x-small')

" linear exponential "
plt.subplot(323)
plt.plot(x2, y2, c='b')
plt.xlabel('Time (years)', size='x-small')
plt.ylabel('Fraction Remaining', size='x-small')
plt.title('Exponential Decay of C-14', size='x-small')
plt.xlim(0, 28650)
plt.yscale('log')

" linear two lines "
plt.subplot(324)
plt.plot(x3, y31, 'r--', x3, y32, 'g')
plt.xlabel('Time (years)', size='x-small')
plt.ylabel('Fraction Remaining', size='x-small')
plt.title('Exponential Decay of Radioactive Elements', size='x-small')
plt.xlim(0, 20000)
plt.ylim(0, 1)
plt.legend(['C-14', 'Ra-226'], fontsize='x-small')

" bar "
plt.subplot(313)
xvals = np.arange(0, 110, 10)
plt.hist(student_grades, facecolor='b', edgecolor='black', bins=xvals)
plt.xticks(np.arange(0, 110, 10))
plt.xlim(0, 100)
plt.xlabel('Grades', size='x-small')
plt.ylabel('Number of Students', size='x-small')
plt.title('Project A', size='x-small')

plt.subplots_adjust(wspace=0.5, hspace=1)
plt.suptitle('All in One')
plt.show()
