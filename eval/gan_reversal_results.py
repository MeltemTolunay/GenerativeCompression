from matplotlib.image import imread
import matplotlib.pyplot as plt
import random
import numpy as np
import os
from PIL import Image
import collections
import pickle
import tensorflow as tf
import argparse


def print_mean_metrics(metrics, name):
    for x in metrics.keys():
        print('name: mean {}={}'.format(x,sum(metrics[x])/len(metrics[x])))

def reshape_to_square(png_image_filename,new_size):                                            
    image = Image.open(png_image_filename)
    new_image = image.resize((new_size, new_size))
    new_image.save('{}_{}'.format(new_size,png_image_filename))

metrics = collections.defaultdict(list)
sess = tf.InteractiveSession()
image_to_gan_reversed_dict={}
for i in range(10):
        image_to_gan_reversed_dict['original_128_{}.png'.format(i)] = 'reversed_gan_{}.png'.format(i)

for image in image_to_gan_reversed_dict:
    im = Image.open(image)
    A = np.asarray(im, dtype=np.uint8)
    im_compressed = Image.open(image_to_gan_reversed_dict[image])
    im_compressed = im_compressed.convert('RGB')
    A_compressed = np.asarray(im_compressed, dtype=np.uint8)
    print(A_compressed.shape)
    # plt.imshow(A_compressed)
    # plt.show()
    # break
    ########################    METRICS     #######################
    im1 = tf.image.convert_image_dtype(A, tf.float32)
    im2 = tf.image.convert_image_dtype(A_compressed, tf.float32)
    psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
    ssim2 = tf.image.ssim(im1, im2, max_val=1.0)
    #mssim2 = tf.image.ssim_multiscale(im1, im2, max_val=1.0)
    mse2 = tf.reduce_mean(tf.squared_difference(
        tf.cast(A, tf.float32), tf.cast(A_compressed, tf.float32)))

    x1, x2, x3 = sess.run([psnr2, ssim2, mse2])
    #x1, x2, x3, x4 = sess.run([psnr2, ssim2, mse2, mssim2])
    metrics['psnr'].append(x1)
    metrics['ssim'].append(x2)
    metrics['mse'].append(x3)
    #metrics['mssim'].append(x4)
    print(metrics)

print_mean_metrics(metrics, 'gan_reversal')

