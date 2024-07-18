import numpy as np
a = np.matrix('1,2,3;3,4,5')
print(a)
def swap_rows(M,a,b):
    M[a],M[b] = M[b],M[a].copy()
def swap_cols(M,a,b):
    for i in M:
        # i[a], i[b] = i[b],i[a].copy()
        print(i)
swap_cols(a,0,1)
print(a[0][0])