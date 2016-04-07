import sys

import matplotlib.pyplot as plt
import numpy

a = numpy.load(sys.argv[1])

conv1 = a['arr_0']

fig, ax = plt.subplots(8, 8,
                       figsize=(8, 8),
                       dpi=100,
                       squeeze=False)

"""
for i in range(conv1.shape[3]):
    ax[i // 8, i % 8].imshow(conv1[:, :, 0, i], cmap='Greys')
    
"""
conv2 = a['arr_2']
for i in range(min(8, conv2.shape[3])):
    for j in range(min(8, conv2.shape[2])):
        ax[j, i].imshow(conv2[:, :, j, i], cmap='Greys')

fig.savefig(sys.argv[2], dpi=30.)

