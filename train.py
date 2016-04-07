import glob
import itertools
import random
import sys

import cv2
import numpy
import tensorflow as tf

import common
import gen

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W, stride=(1, 1)):
  return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                      padding='SAME')


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def detector_model():
    x = tf.placeholder(tf.float32, [None, 128 * 64])

    W_conv1 = weight_variable([11, 11, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,64,128,1])
    h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1)

    W_conv2 = weight_variable([1, 1, 32, 1])
    b_conv2 = bias_variable([1])
    h_conv2 = tf.nn.sigmoid(conv2d(h_conv1, W_conv2) + b_conv2)

    b_final = bias_variable([1])
    y = tf.reduce_sum(tf.reshape(h_conv2, [-1, 128 * 64]), 1) + b_final

    return x, y, [W_conv1, b_conv1, W_conv2, b_conv2, b_final]


def deep_model():
    x = tf.placeholder(tf.float32, [None, 128 * 64])

    # First layer
    W_conv1 = weight_variable([5, 5, 1, 48])
    b_conv1 = bias_variable([48])
    x_image = tf.reshape(x, [-1,64,128,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    # Second layer
    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 1))

    # Third layer
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

    # Densely connected layer
    W_fc1 = weight_variable([32 * 8 * 128, 2048])
    b_fc1 = bias_variable([2048])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 32 * 8 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Output layer
    W_fc2 = weight_variable([2048, 7 * len(common.CHARS)])
    b_fc2 = bias_variable([7 * len(common.CHARS)])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2
    y = tf.reshape(y, [-1, 7, len(common.CHARS)])

    # Presence indicator
    W_p = weight_variable([2048, 1])
    b_p = bias_variable([1])
    p = tf.matmul(h_fc1, W_p) + b_p

    y = tf.concat(1, [p, tf.reshape(y, [-1, 7 * len(common.CHARS)])])

    return (x, y, [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3,
                     W_fc1, b_fc1, W_fc2, b_fc2, W_p, b_p],
            [x_image, h_pool1, h_pool2, h_pool3])


def im_to_vec(im):
    return im.flatten()


def code_to_vec(p, code):
    def char_to_vec(c):
        y = numpy.zeros((len(common.CHARS),))
        y[common.CHARS.index(c)] = 1.0
        return y

    c = numpy.vstack([char_to_vec(c) for c in code])

    return numpy.concatenate([[1. if p else 0], c.flatten()])


def read_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        code = fname.split("/")[1][9:16]
        p = fname.split("/")[1][17] == '1'
        yield im_to_vec(im), code_to_vec(p, code)


def read_detect_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        p = fname.split("/")[1][9]
        yield im_to_vec(im), 0. if p == '0' else 1.


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


def batch(it, batch_size):
    out = []
    for x in it:
        out.append(x)
        if len(out) == batch_size:
            yield out
            out = []
    if out:
        yield out


def read_batches(batch_size):
    def gen_vecs():
        for im, c, p in gen.generate_ims(batch_size, bg_prob=0.0):
            yield im_to_vec(im), code_to_vec(p, c)
    while True:
        yield unzip(gen_vecs())


def read_detect_batches(batch_size, bg_prob=0.0):
    def gen_vecs():
        for im, c, p in gen.generate_ims(batch_size, bg_prob=bg_prob):
            yield im_to_vec(im), 1. if p else 0.
    while True:
        yield unzip(gen_vecs())


def train_detector(learn_rate, report_steps, batch_size, bg_prob,
                   initial_weights=None):
    x, y, params = detector_model()

    y_ = tf.placeholder(tf.float32, [None])

    #cross_entropy = (tf.nn.sigmoid_cross_entropy_with_logits(y, y_) +
    #                 tf.nn.sigmoid_cross_entropy_with_logits(1. - y, 1 - y_))
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y, y_)
    #cross_entropy = tf.reduce_sum((y_ * -tf.log(tf.nn.sigmoid(y)) +
    #                              (1 - y_) * -tf.log(1 - tf.nn.sigmoid(y))))
    #cross_entropy = tf.reduce_sum(tf.maximum(y, 0) -
    #                              y * y_ +
    #                              tf.log(1 + tf.exp(-tf.abs(y))))
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

    result = tf.greater(y, 0.0)

    init = tf.initialize_all_variables()

    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    def do_report():
        r, s, c = sess.run([result, tf.nn.sigmoid(y), cross_entropy],
                           feed_dict={x: test_xs, y_: test_ys})

        print "".join("{:4d}".format(int(100 * x)) for x in s)
        print "".join("{:4d}".format(int(100 * x)) for x in test_ys)

        print numpy.sum(c)

        false_positives = (numpy.sum(r * (1. - test_ys)) /
                            numpy.sum((1. - test_ys)))
        false_negatives = (numpy.sum((1. - r) * test_ys) /
                            numpy.sum(test_ys))

        print "B{:3d} fp:{:2.02f}% fn:{:2.02f}%".format(
            batch_idx, 100. * false_positives, 100. * false_negatives)

    def do_batch():
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if batch_idx % report_steps == 0:
            do_report()

    with tf.Session() as sess:
        sess.run(init)
        if initial_weights is not None:
            sess.run(assign_ops)

        test_xs, test_ys = unzip(list(
                                 read_detect_data("detect_test/*.png"))[:50])

        try:
            batch_iter = enumerate(read_detect_batches(batch_size,
                                                       bg_prob=bg_prob))
            last_weights = None
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                weights = [p.eval() for p in params]
                if all(numpy.all(numpy.isnan(w)) for w in weights):
                    raise WeightsWentNan
                #if last_weights is not None:
                    #print numpy.array(weights) - numpy.array(last_weights)
                last_weights = weights
                do_batch()
        except KeyboardInterrupt, WeightsWentNan:
            numpy.savez("detector_weights.npz", *last_weights)



def train_reader(learn_rate, report_steps, batch_size, initial_weights=None):
    x, y, params, debug_vars = deep_model()

    y_ = tf.placeholder(tf.float32, [None, 7 * len(common.CHARS) + 1])

    digits_loss = tf.nn.softmax_cross_entropy_with_logits(
                                          tf.reshape(y[:, 1:],
                                                     [-1, len(common.CHARS)]),
                                          tf.reshape(y_[:, 1:],
                                                     [-1, len(common.CHARS)]))
    digits_loss = tf.reduce_sum(digits_loss)
    presence_loss = 10. * tf.nn.sigmoid_cross_entropy_with_logits(
                                                          y[:, :1], y_[:, :1])
    presence_loss = tf.reduce_sum(presence_loss)
    cross_entropy = digits_loss + presence_loss
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

    best = tf.argmax(tf.reshape(y[:, 1:], [-1, 7, len(common.CHARS)]), 2)
    correct = tf.argmax(tf.reshape(y_[:, 1:], [-1, 7, len(common.CHARS)]), 2)

    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    init = tf.initialize_all_variables()

    def vec_to_plate(v):
        return "".join(common.CHARS[i] for i in v)

    def do_report():
        r = sess.run([best,
                      correct,
                      tf.greater(y[:, 0], 0),
                      y_[:, 0],
                      digits_loss,
                      presence_loss,
                      cross_entropy],
                     feed_dict={x: test_xs, y_: test_ys})
        num_correct = numpy.sum(
                        numpy.logical_or(
                            numpy.all(r[0] == r[1], axis=1),
                            numpy.logical_and(r[2] < 0.5,
                                              r[3] < 0.5)))
        r_short = (r[0][:190], r[1][:190], r[2][:190], r[3][:190])
        for b, c, pb, pc in zip(*r_short):
            print "{} {} <-> {} {}".format(vec_to_plate(c), pc,
                                           vec_to_plate(b), float(pb))
        num_p_correct = numpy.sum(r[2] == r[3])

        print ("B{:3d} {:2.02f}% {:02.02f}% loss: {} "
               "(digits: {}, presence: {}) |{}|").format(
            batch_idx,
            100. * num_correct / (len(r[0])),
            100. * num_p_correct / len(r[2]),
            r[6],
            r[4],
            r[5],
            "".join("X "[numpy.array_equal(b, c) or (not pb and not pc)]
                                           for b, c, pb, pc in zip(*r_short)))

    def do_batch():
        sess.run(train_step,
                      feed_dict={x: batch_xs, y_: batch_ys})
        if batch_idx % report_steps == 0:
            do_report()

    with tf.Session() as sess:
        sess.run(init)
        if initial_weights is not None:
            sess.run(assign_ops)

        test_xs, test_ys = unzip(list(read_data("test/*.png"))[:50])

        try:
            batch_iter = enumerate(read_batches(batch_size))
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                weights = [p.eval() for p in params]
                if all(numpy.all(numpy.isnan(w)) for w in weights):
                    raise WeightsWentNan
                last_weights = weights
                do_batch()
        except KeyboardInterrupt, WeightsWentNan:
            numpy.savez("weights.npz", *last_weights)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        f = numpy.load(sys.argv[2])
        initial_weights = [f[n] for n in sorted(f.files,
                                                key=lambda s: int(s[4:]))]
    else:
        initial_weights = None

    if sys.argv[1] == "detect":
        train_detector(learn_rate=0.1,
                       report_steps=10,
                       batch_size=10,
                       bg_prob=0.5,
                       initial_weights=initial_weights)
    elif sys.argv[1] == "read":
        train_reader(learn_rate=0.0001,
                     report_steps=20,
                     batch_size=50,
                     initial_weights=initial_weights)

