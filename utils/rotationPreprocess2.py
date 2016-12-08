import tensorflow as tf
import numpy as np

def rotationPreprocess2(n,np):
    np = int(np)
    cos_list = [[[-1,-1] for i in range(n)] for j in range(n-1)]
    sin_list = [[[-1,-1] for i in range(n)] for j in range(n-1)]
    nsin_list = [[[-1,-1] for i in range(n)] for j in range(n-1)]
    cos_idxs = [[-1 for i in range(n)] for j in range(n-1)]
    sin_idxs = [[-1 for i in range(n)] for j in range(n-1)]
    nsin_idxs = [[-1 for i in range(n)] for j in range(n-1)]

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
            cos_list[idx][i] = [i,i]
            cos_list[idx][j] = [j,j]
            sin_list[idx][i] = [i,j]
            nsin_list[idx][j] = [j,i]
            cos_idxs[idx][i] = theta_num
            cos_idxs[idx][j] = theta_num
            sin_idxs[idx][i] = theta_num
            nsin_idxs[idx][j] = theta_num
            theta_num += 1
        head = (head + 2) % (n - 1)
        tail = (tail + 1) % (n - 1)
        theta_idx += 1
        start_idx = (start_idx + 2) % (n - 1)
        idx = start_idx




    sin_list = [[i for i in j if i != [-1,-1]] for j in sin_list]
    nsin_list = [[i for i in j if i != [-1,-1]] for j in nsin_list]
    sin_idxs = [[i for i in j if i != -1] for j in sin_idxs]
    nsin_idxs = [[i for i in j if i != -1] for j in nsin_idxs]

    # print("cos_list")
    # print(cos_list)
    # print("sin_list")
    # print(sin_list)
    # print("nsin_list")
    # print(nsin_list)
    # print("cos_idxs")
    # print(cos_idxs)
    # print("sin_idxs")
    # print(sin_idxs)
    # print("nsin_idxs")
    # print(nsin_idxs)

    cos_list = [tf.constant(j, dtype=tf.int64) for j in cos_list]
    sin_list = [tf.constant(j, dtype=tf.int64) for j in sin_list]
    nsin_list = [tf.constant(j, dtype=tf.int64) for j in nsin_list]
    cos_idxs = [tf.constant(j, dtype=tf.int64) for j in cos_idxs]
    sin_idxs = [tf.constant(j, dtype=tf.int64) for j in sin_idxs]
    nsin_idxs = [tf.constant(j, dtype=tf.int64) for j in nsin_idxs]

    return cos_list, sin_list, nsin_list, cos_idxs, sin_idxs, nsin_idxs
