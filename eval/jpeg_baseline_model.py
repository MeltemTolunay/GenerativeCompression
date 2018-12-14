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
#from skimage.measure import compare_ssim, compare_mse, compare_psnr


DATA_DIRECTORY = 'test_set'
q=10
metrics = collections.defaultdict(list)
sess = tf.InteractiveSession()

i=-1
for file in os.listdir(DATA_DIRECTORY):
    i+=1
    filename = os.fsdecode(file)
    if filename.endswith('.png'):
        pic_path = os.path.join('test_set', filename)
        im = Image.open(pic_path)
        A = np.asarray(im, dtype=np.uint8)
        print(A.shape)
        # plt.imshow(A)
        # plt.show()
        #im = Image.open(os.path.join(DATA_DIRECTORY, filename))
        IMAGE_JPEG_1 = os.path.join(DATA_DIRECTORY,'{}_{}.jpeg'.format(i, q))
        im.save(IMAGE_JPEG_1, "JPEG", quality=q, optimize=True)
        im_compressed = Image.open(IMAGE_JPEG_1)
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


# im1 = Image.open("1.png")
# im1 = im1.convert('RGB')

