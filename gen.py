#!/usr/bin/env python
#
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



"""
Generate training and test images.

"""


__all__ = (
    'generate_ims',
)


import math
import os
import random
import sys

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common

FONT_PATH = "UKNumberPlate.ttf"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized

DETECT_OUTPUT_SHAPE = (64, 128)
READ_OUTPUT_SHAPE = (64, 128)

CHARS = common.CHARS + " "


def make_char_ims(output_height):
    font_size = output_height * 4

    font = ImageFont.truetype(FONT_PATH, font_size)

    height = max(font.getsize(c)[1] for c in CHARS)

    for c in CHARS:
        width = font.getsize(c)[0]
        im = Image.new("RGBA", (width, height), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw.text((0, 0), c, (255, 255, 255), font=font)
        scale = float(output_height) / height
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


def get_transformed_shape_size(M, shape):
    """Return the size of the bounding box of a transformed shape."""
    h, w = shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    transformed_size = numpy.array(numpy.max(M * corners, axis=1) -
                                   numpy.min(M * corners, axis=1)).flatten()
    return transformed_size


def make_transform(yaw, pitch, roll, from_shape, bounds):
    """
    Make a 2x2 transform from the given parameters.

    :param yaw:
        Yaw angle to rotate by.

    :param pitch:
        Pitch angle to rotate by.

    :param roll:
        Roll angle to rotate by.

    :param from_shape:
        Shape of the image being tranformed.

    :param bounds:
        The scale will be selected such that the resulting shape's size is
        within these bounds.

    :return:
        A tuple `M`, `size` where `M` is the transformation, and `size` is the
        size of the bounding box containing the transformed shape.

    """
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    skewed_size = get_transformed_shape_size(M, from_shape)
    scale = numpy.min(numpy.array(bounds) / skewed_size)

    return M * scale, skewed_size * scale


def make_affine_transform(from_shape, to_shape, 
                          yaw_range, pitch_range, roll_range, scale_range,
                          completely_inside=False):
    """
    Make a random affine transform for a shape, to fit within an output image. 

    A rotation (specified in terms of yaw/pitch/roll) are selected based on
    ranges specified in the arguments.
    
    Scale is similarly specified by a range. A scale of 1.0 corresponds with an
    output image whose size equals that of the output image in exactly one
    dimension, with the other dimension being smaller.
    
    If `on_edge` is `False` translation is selected uniformly with the
    constraint that the transformed shape's bounding box lies entirely within
    the output shape. If `on_edge` is `True` translation is selected uniformly
    with the constraint that the transformed shape's bounding box intersects
    with one or more edges of the output shape.

    :param from_shape:
        The shape being transformed.

    :param to_shape:
        The shape of the output image.

    :param yaw_range:
        A (min, max) tuple defining the uniform distribution from which the yaw
        is selected.

    :param pitch_range:
        A (min, max) tuple defining the uniform distribution from which the
        pitch is selected.

    :param roll_range:
        A (min, max) tuple defining the uniform distribution from which the
        roll is selected.

    :param scale_range:
        A (min, max) tuple defining the uniform distribution from which the
        scale is selected. The maximum must be less than 1.0.

    :param completely_inside:
        Indicate whether the bounding box of the transformed shape's bounding
        box should lie entirely within the output shape (`True`) or whether
        only part of the part of the transformed shape's bounding box should
        lie within the output shape (`False`).

    :return:
        A tuple `M`, `out_of_bounds`, `scale` where `M` is the 2x3 affine
        transform described above, `out_of_bounds` indicates whether the
        transformed shape's bounding box partially lies outside of the output
        shape (note `out_of_bounds` is always `False` if `completely_inside` is
        `True`), and `scale` indicates the scale that was chosen.

    """
    yaw = random.uniform(*yaw_range)
    pitch = random.uniform(*pitch_range)
    roll = random.uniform(*roll_range)
    scale = random.uniform(*scale_range)
    bounds = scale * numpy.array([to_shape[1], to_shape[0]])

    M, transformed_size = make_transform(yaw, pitch, roll, from_shape, bounds)

    # Set `t` to the translation which puts the centre of the plate at 0, 0.
    t = M * numpy.matrix([[-from_shape[1], -from_shape[0]]]).T * 0.5

    # Determine out the x and y coordinates of the output shape centre.
    if completely_inside:
        x = random.uniform(transformed_size[0] / 2,
                           to_shape[1] - transformed_size[0] / 2)
        y = random.uniform(transformed_size[1] / 2,
                           to_shape[0] - transformed_size[1] / 2)
        out_of_bounds = False
    else:
        x = random.uniform(-transformed_size[0] / 2,
                           to_shape[1] + transformed_size[0] / 2)
        y = random.uniform(-transformed_size[1] / 2,
                           to_shape[0] + transformed_size[1] / 2)
        out_of_bounds = (x < transformed_size[0] / 2. or
                         x > to_shape[1] - transformed_size[0] / 2 or
                         y < transformed_size[1] / 2. or
                         y > to_shape[0] - transformed_size[1] / 2)

    # Add the above to `t` to get the final translation.
    t += numpy.matrix([[x], [y]])

    return numpy.hstack([M, t]), out_of_bounds, scale
                                         

def generate_code():
    return "{}{}{}{} {}{}{}".format(
        random.choice(common.LETTERS),
        random.choice(common.LETTERS),
        random.choice(common.DIGITS),
        random.choice(common.DIGITS),
        random.choice(common.LETTERS),
        random.choice(common.LETTERS),
        random.choice(common.LETTERS))


def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out


def generate_plate(font_height, char_ims):
    h_padding = random.uniform(0.2, 0.4) * font_height
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(-0.05, 0.05)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors()
    
    text_mask = numpy.zeros(out_shape)
    
    x = h_padding
    y = v_padding 
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")


def generate_bg(num_bg_images, output_shape):
    found = False
    while not found:
        fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255.
        if (bg.shape[1] >= output_shape[1] and
            bg.shape[0] >= output_shape[0]):
            found = True

    x = random.randint(0, bg.shape[1] - output_shape[1])
    y = random.randint(0, bg.shape[0] - output_shape[0])
    bg = bg[y:y + output_shape[0], x:x + output_shape[1]]

    return bg


def generate_detect_im(char_ims, num_bg_images, output_shape):
    output_size = numpy.array([output_shape[1], output_shape[0]])

    bg = generate_bg(num_bg_images, output_shape)

    plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims)
    plate_shape = plate.shape
    plate_size = numpy.array([[plate.shape[1], plate.shape[0]]]).T
    
    M, out_of_bounds, scale = make_affine_transform(plate.shape,
                                                    output_shape,
                                                    roll_range=(-0.3, 0.3),
                                                    pitch_range=(-0.2, 0.2),
                                                    yaw_range=(-1.2, 1.2),
                                                    scale_range=(0.3, 1.0),
                                                    completely_inside=False)

    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = cv2.resize(out, (output_shape[1], output_shape[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    plate_centre = numpy.array(M * numpy.concatenate([plate_size * 0.5,
                                                      [[1.]]])).T[0]
    plate_centre = plate_centre/ output_size
    skewed_size = get_transformed_shape_size(M[:, :2], plate_shape)
    scale = numpy.max(skewed_size / output_size)

    return out, plate_centre, scale


def generate_im(char_ims, num_bg_images, output_shape):
    bg = generate_bg(num_bg_images, output_shape)

    plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims)
    
    M, out_of_bounds, scale = make_affine_transform(plate.shape,
                                                    output_shape,
                                                    roll_range=(-0.3, 0.3),
                                                    pitch_range=(-0.2, 0.2),
                                                    yaw_range=(-1.2, 1.2),
                                                    scale_range=(0.9, 1.0),
                                                    completely_inside=True)

    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = cv2.resize(out, (output_shape[1], output_shape[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    return out, code, True


def generate_ims(num_images, output_shape):
    """
    Generate a number of number plate images.

    :param num_images:
        Number of images to generate.

    :return:
        Iterable of number plate images.

    """
    char_ims = dict(make_char_ims(FONT_HEIGHT))
    num_bg_images = len(os.listdir("bgs"))
    for i in range(num_images):
        yield generate_im(char_ims, num_bg_images, output_shape)


def generate_detect_ims(num_images, output_shape):
    char_ims = dict(make_char_ims(FONT_HEIGHT))
    num_bg_images = len(os.listdir("bgs"))
    for i in range(num_images):
        yield generate_detect_im(char_ims, num_bg_images, output_shape)


if __name__ == "__main__":
    os.mkdir("test")

    os.mkdir("test/read")
    im_gen = generate_ims(int(sys.argv[1]), READ_OUTPUT_SHAPE)
    for img_idx, (im, c, p) in enumerate(im_gen):
        fname = "test/read/{:08d}_{}_{}.png".format(img_idx, c,
                                                    "1" if p else "0")
        print fname
        cv2.imwrite(fname, im * 255.)

    os.mkdir("test/detect")
    im_gen = generate_detect_ims(int(sys.argv[1]), DETECT_OUTPUT_SHAPE)
    for img_idx, (im, centre, scale) in enumerate(im_gen):
        centre_x, centre_y = centre.flatten()
        fname = "test/detect/{:08d}_{:.3f}_{:.3f}_{:.3f}.png".format(img_idx,
                                                                     centre_x,
                                                                     centre_y,
                                                                     scale)
        print fname
        cv2.imwrite(fname, im * 255.)
