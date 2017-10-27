import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n_bins = 50
x = np.random.randn(1000, 3)

print(x[0,:])
values = ['x','y', 'z']
plt.hist(x, n_bins, normed=1, histtype='barstacked', stacked=True,label=values)
plt.legend()
plt.show()
plt.cla()
plt.clf()


plt.hist(x[:,2], n_bins, normed=1, histtype='bar')

plt.show()