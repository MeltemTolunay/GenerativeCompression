from matplotlib.image import imread
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import tensorflow as tf
from PIL import Image
#from skimage import img_as_float64
from skimage.measure import compare_ssim, compare_mse, compare_psnr
# from skimage import io
# io.use_plugin('matplotlib')
import collections
import pickle
import argparse

MAX_ITERATIONS= 30
SAVE_METRICS_ITERATION= 2
DATA_DIRECTORY='test_set'

def dist(x,y):
    return np.linalg.norm(x-y)**2

def mse(x, y):
    return np.mean((x - y)**2)

def kmeans(x,K,eps=1):
    def get_score():
        score=0
        for idx in range(m):
            score+=dist(centroids[cluster_assignments[idx]],x[idx])
        return score
    m, n = x.shape
    indices=np.random.choice(m,K,replace=False)
    centroids = x[indices]
    cluster_assignments=[-1]*m
    new_score = float("+inf")
    iteration=0

    while True:
        iteration+=1
        new_centroids = np.zeros((K,n))
        counts = [0]*K
        old_score=new_score
        for i in range(m):
            ans=min([[dist(x[i],centroids[j]),j] for j in range(K)])
            cluster_assignments[i]=ans[-1]
            new_centroids[ans[-1]] += x[i]
            counts[ans[-1]] += 1
        for i in range(K):
            new_centroids[i] = new_centroids[i]/counts[i]
        centroids=new_centroids
        new_score=get_score()
        print(old_score-new_score, (old_score-new_score<0)*"!!!!!! NEGATIVE, must always be POSTIVE")
        if old_score-new_score < eps or iteration > MAX_ITERATIONS:
            break
    return centroids,cluster_assignments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='K-means Baseline Compression')
    parser.add_argument('K', metavar='K', type=int, nargs='+',
                        help='the number of clusters K desired in K means')
    args = parser.parse_args()
    K=args.K[0]
    #print(K)
    sess = tf.InteractiveSession()
    metrics=collections.defaultdict(list)
    idx=-1
    for filename in os.listdir(DATA_DIRECTORY):
        if filename.endswith('.png'):
            idx+=1
            pic_path = os.path.join(DATA_DIRECTORY, filename)
            A = np.asarray(Image.open(pic_path), dtype=np.uint8)
            #A = imread(pic_path)
            #A=np.array(A)
            #print A[:50, :50, :]
            rows,cols,depth=A.shape
            A_compressed = np.empty(A.shape, dtype=np.uint8)
            # K=16
            #print(rows*cols)
            #A_flattened = [A[i, j, :] for i in range(rows) for j in range(cols)]
            A_flattened=A.reshape(rows*cols,depth)
            #print A_flattened[:50, :50]
            centroids, cluster_assignments = kmeans(A_flattened, K)
            #print centroids, cluster_assignments
            #np.savetxt(os.path.join('output', 'pb5_centroids.txt'), centroids)
            #np.savetxt(os.path.join('output', 'pb5_cluster_assignments.txt'), cluster_assignments)

            for i in range(rows): 
                for j in range(cols):
                    A_compressed[i,j]=centroids[cluster_assignments[i*cols+j]]
            #print A_compressed[:3, :3,:]
            #plt.imshow(A_compressed)
            
            #savepath=os.path.join('output', 'kmeans', '{}_compressed_'.format(K)+filename)
            savepath=os.path.join(DATA_DIRECTORY, 'kmeans_{}_compressed_'.format(K)+filename)
            plt.imsave(savepath, A_compressed)
            
            #Get SSIM score
            ## A_compressed = img_as_float64(savepath, force_copy=True)
            ## A=img_as_float64(pic_path,force_copy=True)
            
            # mse1 = compare_mse(A, A_compressed)
            # ssim1 = compare_ssim(A, A_compressed,multichannel=True,gaussian_weights=True)
            # psnr1 = compare_psnr(A, A_compressed)
            # ssim_arr[idx]=ssim1
            # print(A_compressed.shape, A.shape)
            # print(psnr1,ssim1, mse1)

            im1 = tf.image.convert_image_dtype(A, tf.float32)
            im2 = tf.image.convert_image_dtype(A_compressed, tf.float32)
            psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
            ssim2 = tf.image.ssim(im1, im2, max_val=1.0)
            #mssim2=tf.image.ssim_multiscale(im1, im2, max_val=1.0)
            mse2 = tf.reduce_mean(tf.squared_difference(tf.cast(A,tf.float32), tf.cast(A_compressed,tf.float32)))
            #mse2 = tf.metrics.mean_squared_error(im1, im2)

            # sess.run(tf.global_variables_initializer())
            x1, x2, x3 = sess.run([psnr2, ssim2, mse2])
            #x1, x2, x3,x4 = sess.run([psnr2, ssim2, mse2, mssim2])
            metrics['psnr'].append(x1)
            metrics['ssim'].append(x2)
            metrics['mse'].append(x3)
            #metrics['mssim'].append(x4)
            print(metrics)
            if idx%SAVE_METRICS_ITERATION==0:
                with open(os.path.join(DATA_DIRECTORY, 'metrics_{}.pickle'.format(K)), 'wb') as handle:
                    pickle.dump(metrics, handle)
                print("Succesfully pickled metrics dict")
            #print(pickle.load(open('metrics_{}.pickle'.format(K), "rb")))
            print("------------------------DONE with image {}----------------------------".format(filename))
    
    with open(os.path.join(DATA_DIRECTORY, 'metrics_{}.pickle'.format(K)), 'wb') as handle:
        pickle.dump(metrics, handle)
        print("Succesfully pickled metrics dict")
