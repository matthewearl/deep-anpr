# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

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

