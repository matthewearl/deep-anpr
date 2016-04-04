import sys

import matplotlib.pyplot as plt
import numpy

a = numpy.load(sys.argv[1])

conv1 = a['arr_0']

fig, ax = plt.subplots(8, 8,
                       figsize=(8, 8),
                       dpi=20,
                       squeeze=False)

for i in range(conv1.shape[3]):
    #ax[i // 8, i % 8].imshow(conv1[:, i].reshape((28, 28)), cmap='RdBu')
    ax[i // 8, i % 8].imshow(conv1[:, :, 0, i], cmap='RdBu')
    
"""
conv2 = a['arr_2']
for i in range(conv2.shape[3]):
    for j in range(conv2.shape[2]):
        ax[i, 1 + j].imshow(conv1[:, :, j, i], cmap='hot')
"""

fig.savefig(sys.argv[2])

