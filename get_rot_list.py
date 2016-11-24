def gen_rot_list(n):
    arr = [[0] * n for i in range(n-1)]
    rot_list = [[] for i in range(n-1)]
    idx = 0
    for i in range(n-1):
        for j in range(i+1, n):
            while arr[idx][i] == 1:
                idx = (idx+1) % (n-1)
            arr[idx][i] = 1
            arr[idx][j] = 1
            rot_list[idx].append([i, j])
    return rot_list

print gen_rot_list(10)
