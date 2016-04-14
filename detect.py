import cv2
import numpy
import tensorflow as tf

import model


def make_scaled_ims(im, min_shape):
    yield im
    while True:
        shape = (int(im.shape[0] / math.sqrt(2)),
                 int(im.shape[1] / math.sqrt(2)))
        if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
            break
        im = cv2.resize(im, (shape[1], shape[0]))
        yield im


def detect(im, param_vals):
    # Convert the image to various scales.
    scaled_ims = numpy.array(make_scaled_ims(model.WINDOW_SHAPE))

    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_training_model()

    # Execute the model.
    with tf.Session(config=tf.ConfigProto()) as sess:
        detections = sess.run(y, feed_dict={x: scaled_ims,
                                            params: param_vals})

    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
    for x in numpy.argwhere(detections([:, :, :, 0] > 0.5)):
        img_idx, window_coords = x[0], x[1:]
        img_scale = im.shape[0] / scaled_ims[img_idx].shape[0]

        bbox_centre = window_coords * (8, 4) * img_scale
        bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

        yield bbox_centre - bbox_size / 2, bbox_centre + bbox_size / 2,


if __name__ == "__main__":
    im = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255.

    f = numpy.load(sys.argv[2])
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

    for pt1, pt2 in detect(im, param_vals):
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))

        cv2.rectangle(im, pt1, pt2, 1.0)

    cv2.imwrite("detected.png", im * 255.)

