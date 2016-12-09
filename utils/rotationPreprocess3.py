import tensorflow as tf
import numpy as np

def rotationPreprocess3(n,np):
    np = int(np)
    cos_list = [[] for i in range(np*2)]
    sin_list = [[-1,-1] for i in range(np*2)]
    nsin_list = [[-1,-1] for i in range(np*2)]
    cos_idxs = [-1 for i in range(np*2)]
    sin_idxs = [-1 for i in range(np*2)]
    nsin_idxs = [-1 for i in range(np*2)]

    theta_num = 0

    head = -1
    tail = -1
    arr = [[-1] * n for i in range(n-1)]
    rot_list = [[] for i in range(n-1)]
    start_idx = 0
    idx = 0
    theta_idx = 0
    for i in range(n-1):
        for j in range(i+1, n):
            while arr[idx][i] > -1 or arr[idx][j] > -1:
                idx = (idx+1) % (n-1)
            arr[idx][i] = theta_idx
            arr[idx][j] = theta_idx
            cos_list[idx*n + i] = [i,i]
            cos_list[idx*n + j] = [j,j]
            sin_list[idx*n + i] = [i,j]
            nsin_list[idx*n + j] = [j,i]
            cos_idxs[idx*n + i] = theta_num
            cos_idxs[idx*n + j] = theta_num
            sin_idxs[idx*n + i] = theta_num
            nsin_idxs[idx*n + j] = theta_num
            theta_num += 1
        head = (head + 2) % (n - 1)
        tail = (tail + 1) % (n - 1)
        theta_idx += 1
        start_idx = (start_idx + 2) % (n - 1)
        idx = start_idx

    sin_list = [i for i in sin_list if i != [-1,-1]]
    nsin_list = [i for i in nsin_list if i != [-1,-1]]
    sin_idxs = [i for i in sin_idxs if i != -1]
    nsin_idxs = [i for i in nsin_idxs if i != -1]

    cos_list = tf.constant(cos_list, dtype=tf.int64)
    sin_list = tf.constant(sin_list, dtype=tf.int64)
    nsin_list = tf.constant(nsin_list, dtype=tf.int64)
    cos_idxs = tf.constant(cos_idxs, dtype=tf.int64)
    sin_idxs = tf.constant(sin_idxs, dtype=tf.int64)
    nsin_idxs = tf.constant(nsin_idxs, dtype=tf.int64)

    return cos_list, sin_list, nsin_list, cos_idxs, sin_idxs, nsin_idxs
