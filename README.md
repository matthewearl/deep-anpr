# Deep ANPR

Using neural networks to build an automatic number plate recognition system.
See [this blog post](http://matthewearl.github.io/2016/05/06/cnn-anpr/) for an
explanation.

Usage is as follows:

1. `extractbgs.py SUN397.tar.gz`: Extract ~3GB of background images from the [SUN database](http://groups.csail.mit.edu/vision/SUN/)
   into `bgs/`. (`bgs/` must not already exist.) The tar file (39GB) can be [downloaded here](http://groups.csail.mit.edu/vision/SUN1old/SUN397.tar).

2. `gen.py 1000`: Generate 1000 test set images in `test/`. (`test/` must not
    already exist.) This step requires `UKNumberPlate.ttf` to be in the current
    directory, which can be [downloaded here](http://www.dafont.com/uk-number-plate.font).

3. `train.py`: Train the model. A GPU is recommended for this step.

4. `detect.py in.jpg weights.npz out.jpg`: Detect number plates in an image.

The project has the following dependencies:

* [TensorFlow](https://tensorflow.org)
* OpenCV
* NumPy

