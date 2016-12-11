import tensorflow as tf
import numpy as np

def rotationPreprocess(n, num_rot=None):
    print("in rotation preprocess")
    num_rot = num_rot or n-1
    np = int(n*(n-1)/2*num_rot/(n-1))
    indices = [[-1,-1] for i in range(np*4)]
    values = [-1 for i in  range(np*4)]

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
            if idx >= num_rot:
                continue

            indices[idx*n*2 + 2*i] = [i,i]
            indices[idx*n*2 + 2*j+1] = [j,j]
            indices[idx*n*2 + 2*i+1] = [i,j]
            indices[idx*n*2 + 2*j] = [j,i]

            values[idx*n*2 + 2*i] = theta_num
            values[idx*n*2 + 2*j+1] = theta_num
            values[idx*n*2 + 2*i+1] = np + theta_num
            values[idx*n*2 + 2*j] = np*2 + theta_num

            theta_num += 1
        head = (head + 2) % (n - 1)
        tail = (tail + 1) % (n - 1)
        theta_idx += 1
        start_idx = (start_idx + 2) % (n - 1)
        idx = start_idx

    indices = tf.constant(indices, dtype=tf.int64)
    values = tf.constant(values, dtype=tf.int64)
    print("exiting rotation preprocess")
    return indices, values
