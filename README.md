# Deep ANPR

Using neural networks to build an automatic number plate recognition system.
See [this blog post](http://matthewearl.github.io/2016/04/17/cnn-anpr/) for an
explanation.

Usage is as follows:

* `gen.py 1000`: Generate 1000 test set images in `./test`.
* `train.py`: Train the model.
* `detect.py in.jpg weights.npz out.jpg`: Detect number plates in an image.

The project has the following dependencies:

* [TensorFlow](https://tensorflow.org)
* OpenCV
* NumPy

