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

__all__ = (
    'generate_ims',
    'extract_backgrounds',
)

import math
import random
import sys
import tarfile

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common

FONT_PATH = "/media/vbox_google_drive/UKNumberPlate.ttf"
FONT_HEIGHT = 32  # Pixel size to which the chars are resized

OUTPUT_SHAPE = (64, 128)

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


def make_affine_transform(center_from, center_to, translation_amount,
                          min_scale, max_scale):
    scale = random.uniform(min_scale, max_scale)
    roll = random.uniform(-0.15, 0.15)
    pitch = random.uniform(-0.2, 0.2)
    yaw = random.uniform(-0.9, 0.9)

    scale = max_scale
    translation_amount = roll = pitch = yaw = 0.

    trans = numpy.matrix([random.uniform(-translation_amount,
                                         translation_amount),
                          random.uniform(-translation_amount,
                                         translation_amount)]).T

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M


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


def generate_im(char_ims):
    #bg = cv2.imread("bgs/{:08d}.jpg".format(random.randint(0, 108600)),
    #               cv2.CV_LOAD_IMAGE_GRAYSCALE) / 255.
    #bg = bg[bg.shape[0] // 4:3 * (bg.shape[0] // 4), :]
    bg = numpy.random.random(OUTPUT_SHAPE)

    plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims)
    
    max_scale = 0.8 * bg.shape[1] / plate.shape[1]
    M = make_affine_transform(
            center_from=numpy.matrix([plate.shape[1], plate.shape[0]]).T * 0.5,
            center_to=numpy.matrix([bg.shape[1], bg.shape[0]]).T * 0.5,
            min_scale=0.8 * max_scale,
            max_scale=max_scale,
            translation_amount=bg.shape[1] // 10)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate * plate_mask + bg * (1.0 - plate_mask)

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    return out, code


def generate_ims(num_images):
    """
    Generate a number of number plate images.

    """
    char_ims = dict(make_char_ims(FONT_HEIGHT))
    for i in range(num_images):
        yield generate_im(char_ims)


def im_from_file(f):
    a = numpy.asarray(bytearray(f.read()), dtype=numpy.uint8)
    return cv2.imdecode(a, cv2.CV_LOAD_IMAGE_GRAYSCALE)


def extract_backgrounds(archive_name):
    """
    Extract backgrounds from provided tar archive.

    JPEGs from the archive are converted into grayscale, and cropped/resized to
    256x256, and saved in ./bgs/.

    """
    t = tarfile.open(name=archive_name)

    def members():
        m = t.next()
        while m:
            yield m
            m = t.next()
    index = 0
    for m in members():
        if not m.name.endswith(".jpg"):
            continue
        f =  t.extractfile(m)
        try:
            im = im_from_file(f)
        finally:
            f.close()
        if im is None:
            continue
        
        if im.shape[0] > im.shape[1]:
            im = im[:im.shape[1], :]
        else:
            im = im[:, :im.shape[0]]
        if im.shape[0] > 256:
            im = cv2.resize(im, (256, 256))
        fname = "bgs/{:08}.jpg".format(index)
        print fname
        cv2.imwrite(fname, im)
        index += 1


if __name__ == "__main__":
    for img_idx, (im, c) in enumerate(generate_ims(int(sys.argv[1]))):
        fname = "test/{:08d}_{}.png".format(img_idx, c)
        print fname
        cv2.imwrite(fname, im * 255.)

