import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt

data = genfromtxt('data.csv', delimiter=',')


matplotlib.rcParams.update({'font.size': 16})
fig, ax1 = plt.subplots()
lns1 = ax1.plot(data[:,0], data[:,2], color='dodgerblue', linewidth=2.5, label='Val Acc')
ax1.set_xlabel('epochs')
ax1.set_ylabel('Validation Accuracy (%)')
plt.grid()

ax2 = ax1.twinx()
lns2 = ax2.plot(data[:,0], data[:,1], color='orange', linewidth=2.5, label='Val Loss')
ax2.set_ylabel('Validation Loss')

lns = lns1+lns2
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc='center right')

fig.tight_layout()

#plt.show()
plt.savefig('foo.png')
