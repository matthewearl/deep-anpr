import math
import sys

import cv2
import numpy
import tensorflow as tf

import common
import model


def make_scaled_ims(im, min_shape):
    ratio = 1. / 2 ** 0.5
    shape = (im.shape[0] * 2, im.shape[1] * 2)

    while True:
        shape = (int(shape[0] * ratio), int(shape[1] * ratio))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        yield cv2.resize(im, (shape[1], shape[0]))


def detect(im, param_vals):
    # Convert the image to various scales.
    scaled_ims = list(make_scaled_ims(im, model.WINDOW_SHAPE))

    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_detect_model()

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        y_vals = []
        for scaled_im in scaled_ims:
            feed_dict = {x: numpy.stack([scaled_im])}
            feed_dict.update(dict(zip(params, param_vals)))
            y_vals.append(sess.run(y, feed_dict=feed_dict))

    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    for i, (scaled_im, y_val) in enumerate(zip(scaled_ims, y_vals)):
        for window_coords in numpy.argwhere(y_val[0, :, :, 0] >
                                                       -math.log(1./0.99 - 1)):
            letter_probs = (y_val[0,
                                  window_coords[0],
                                  window_coords[1], 1:].reshape(
                                    7, len(common.CHARS)))
            letter_probs = common.softmax(letter_probs)

            img_scale = float(im.shape[0]) / scaled_im.shape[0]

            bbox_tl = window_coords * (8, 4) * img_scale
            bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

            yield bbox_tl, bbox_tl + bbox_size


if __name__ == "__main__":
    im = cv2.imread(sys.argv[1])
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.

    f = numpy.load(sys.argv[2])
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

    for pt1, pt2 in detect(im_gray, param_vals):
        pt1 = tuple(reversed(map(int, pt1)))
        pt2 = tuple(reversed(map(int, pt2)))

        color = (0.0, 255.0, 0.0)
        cv2.rectangle(im, pt1, pt2, color)

    cv2.imwrite(sys.argv[3], im)

