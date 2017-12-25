#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt


def trans_img(data):
    # transfer a img   transform data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip()])
    data = transform(data)
    # data=data.numpy()
    data = jitter(data)
    data = whiter(data)

    return data


def jitter(image):
    # image = Image.fromarray(data)
    random_factor = np.random.randint(0, 31) / 10.
    image = ImageEnhance.Color(image).enhance(random_factor)
    random_factor = np.random.randint(5, 11) / 10.
    image = ImageEnhance.Brightness(image).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.
    image = ImageEnhance.Contrast(image).enhance(random_factor)

    image = np.array(image)
    return image


def whiter(image):
    """

    :param image: image.shape=[300,300,3];height,width,channel
    :return: the data after whitering
    """
    im = image.astype('float64')
    data = np.zeros(image.shape).astype('float16')
    w, h, c = image.shape
    for i in range(c):
        data[:, :, i] = (im[:, :, i] - np.mean(im[:, :, i])) / np.std(im[:, :, i])
    return data


def show_data(image):
    if image.dtype != np.uint8:
        data, _ = transfer(image)
    else:
        data = image
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # print colors[1]
    plt.imshow(data)


def transfer(image):
    """
    data is transfered to 0-255,for show
    transfer() & re_transfer() will be used in PIL.Image
    PIL.image was used in resize() & jitter()
    :param image:
    :return: its range of every channel
    """
    data = image.copy()
    min_max = []  # range of channls
    h, w, c = data.shape
    for i in range(c):
        min_d, max_d = (np.min(data[:, :, i]), np.max(data[:, :, i]))
        min_max.append((min_d, max_d))
        zone = max_d - min_d
        if zone < 0.1:
            zone = 0.1
        data[:, :, i] = 1.0 * (data[:, :, i] - min_d) / zone * 255
    data = data.astype('uint8')
    return data, min_max


if __name__ == '__main__':
    pass
