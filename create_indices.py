
g = [[] for i in range(n)]

k = 0
for i in range(1,n+1):
    g[k].append((0,i))
    k += 1

k = 2
steps = 2

for i in range(1,n):
    for j in range(i+1, n+1):
        g[k].append((i,j))
        if j == n-1:
            k = (k+steps)%n
            steps += 1
        elif j == n:
            k = (k+3)%n
        else:
            k = (k+1)%n
print g
