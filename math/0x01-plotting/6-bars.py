#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))


x = ('Farrah', 'Fred', 'Felicia')

plt.bar(x, fruit[0], color='red', width=0.5)
plt.bar(x, fruit[1], bottom=fruit[0], color='yellow', width=0.5)
plt.bar(x, fruit[2], bottom=np.array(fruit[0])+np.array(fruit[1]),
        color='orange', width=0.5)
plt.bar(x, fruit[3], bottom=np.array(fruit[0]) + np.array(fruit[1])
        + np.array(fruit[2]), color='#ffe5b4', width=0.5)
plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 81, 10))
plt.legend(labels=['apples', 'bananas', 'oranges', 'peaches'])
plt.show()
