import glob
import itertools
import random

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


def deep_model2():
    x = tf.placeholder(tf.float32, [None, 128 * 64])

    # First layer
    W_conv1 = weight_variable([5, 5, 1, 24])
    b_conv1 = bias_variable([24])
    x_image = tf.reshape(x, [-1,64,128,1])
    h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(8, 8), stride=(8, 8))

    # Output layer
    W_fc1 = weight_variable([16 * 8 * 24, 7 * len(common.CHARS)])
    b_fc1 = bias_variable([7 * len(common.CHARS)])

    h_pool1_flat = tf.reshape(h_pool1, [-1, 16 * 8 * 24])
    y = tf.matmul(h_pool1_flat, W_fc1) + b_fc1

    y = tf.reshape(y, [-1, len(common.CHARS)])

    return x, y, [W_conv1, b_conv1, W_fc1, b_fc1]


def deep_model1():
    x = tf.placeholder(tf.float32, [None, 128 * 64])

    # First layer
    W_conv1 = weight_variable([5, 5, 1, 24])
    b_conv1 = bias_variable([24])
    x_image = tf.reshape(x, [-1,64,128,1])
    h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(8, 8), stride=(8, 8))

    # Densely connected layer
    W_fc1 = weight_variable([16 * 8 * 24, 1024])
    b_fc1 = bias_variable([1024])

    h_pool1_flat = tf.reshape(h_pool1, [-1, 16 * 8 * 24])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

    # Output layer
    W_fc2 = weight_variable([1024, 7 * len(common.CHARS)])
    b_fc2 = bias_variable([7 * len(common.CHARS)])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    y = tf.reshape(y, [-1, len(common.CHARS)])

    return x, y, [W_conv1, b_conv1,
                  W_fc1, b_fc1, W_fc2, b_fc2]


def deep_model():
    x = tf.placeholder(tf.float32, [None, 128 * 64])

    # First layer
    W_conv1 = weight_variable([5, 5, 1, 24])
    b_conv1 = bias_variable([24])
    x_image = tf.reshape(x, [-1,64,128,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, stride=(2, 2))

    # Second layer
    W_conv2 = weight_variable([5, 5, 24, 32])
    b_conv2 = bias_variable([32])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, stride=(2, 1))

    # Third layer
    W_conv3 = weight_variable([5, 5, 32, 64])
    b_conv3 = bias_variable([64])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, stride=(2, 2))

    # Densely connected layer
    W_fc1 = weight_variable([32 * 8 * 64, 2048])
    b_fc1 = bias_variable([2048])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 32 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Output layer
    W_fc2 = weight_variable([2048, 7 * len(common.CHARS)])
    b_fc2 = bias_variable([7 * len(common.CHARS)])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2
    y = tf.reshape(y, [-1, 7, len(common.CHARS)])

    return x, y, [W_conv1, b_conv1, W_conv2, b_conv2,
                  W_fc1, b_fc1, W_fc2, b_fc2]


MODEL = deep_model1
#LEARN_RATE = 2 * 1e-4
LEARN_RATE = 0.01
BATCH_SIZE = 10
REPORT_STEPS = 10


def im_to_vec(im):
    return im.flatten()


def code_to_vec(code):
    def char_to_vec(c):
        y = numpy.zeros((len(common.CHARS),))
        y[common.CHARS.index(c)] = 1.0
        return y

    return numpy.vstack([char_to_vec(c) for c in code])


def read_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255.
        code = fname.split("/")[1][9:16]
        yield im_to_vec(im), code_to_vec(code)


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
        for im, c in gen.generate_ims(batch_size):
            yield im_to_vec(im), code_to_vec(c)
    while True:
        yield unzip(gen_vecs())


def train(learn_rate):
    x, y, params = MODEL()

    y_ = tf.placeholder(tf.float32, [None, 7, len(common.CHARS)])

    #cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                                       tf.reshape(y, [-1, len(common.CHARS)]),
                                       tf.reshape(y_, [-1, len(common.CHARS)]))
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

    best = tf.argmax(tf.reshape(y, [-1, 7, len(common.CHARS)]), 2)
    correct = tf.argmax(tf.reshape(y_, [-1, 7, len(common.CHARS)]), 2)
    init = tf.initialize_all_variables()

    def vec_to_plate(v):
        return "".join(common.CHARS[i] for i in v)

    def do_report():
        r = sess.run([best, correct],
                     feed_dict={x: test_xs, y_: test_ys})
        num_correct = numpy.sum(r[0] == r[1])
        r_short = (r[0][:190], r[1][:190])
        for b, c in zip(*r_short):
            print "{} <-> {}".format(vec_to_plate(c), vec_to_plate(b))

        print "B{:3d} {:2.02f}% |{}|".format(
            batch_idx,
            100. * num_correct / (7 * len(r[0])),
            "".join("X "[numpy.array_equal(b, c)] for b, c in zip(*r_short)))
        numpy.savez("batches/batch_{}.npz".format(batch_idx),
                    *(p.eval() for p in params))

    def do_batch():
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if batch_idx % REPORT_STEPS == 0:
            do_report()

    with tf.Session() as sess:
        sess.run(init)

        test_xs, test_ys = unzip(list(read_data("test/*.png"))[:50])

        try:
            batch_iter = enumerate(read_batches(BATCH_SIZE))
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                weights = [p.eval() for p in params]
                if all(numpy.all(numpy.isnan(w)) for w in weights):
                    raise WeightsWentNan
                last_weights = weights
                do_batch()
        except KeyboardInterrupt, WeightsWentNan:
            numpy.savez("weights.npz", *last_weights)


if __name__ == "__main__":
    train(LEARN_RATE)

