# %%
import numpy as np

# %%
def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

# # %%
# shape = 100
# b=np.random.rand(shape)
# a=np.random.rand(shape-1)
# c=np.random.rand(shape-1)

# # %%
# A = tridiag(a,b,c)
# A

# %%
def thomas(A, d):
    n = A.shape[0]
    a = np.diag(A, k=-1) # a的下标[1,n)
    a = np.insert(a, 0, 0)
    b = np.diag(A, k=0)  # b的下标[0,n)
    c = np.diag(A, k=1)  # c的下标[0,n-1)
    c_prime = [0 for i in range(n-1)]
    d_prime = [0 for i in range(n)]
    for i in range(n-1):
        if i == 0:
            c_prime[i] = c[i]/b[i]
        else:
            c_prime[i] = c[i]/(b[i]-a[i]*c_prime[i-1])
            
    for i in range(n):
        if i == 0:
            d_prime[i] = d[i]/b[i]
        else:
            d_prime[i] = (d[i]-a[i]*d_prime[i-1])/(b[i]-a[i]*c_prime[i-1])

    # print(c_prime, d_prime)
    x = [0 for i in range(n)]
    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i]-c_prime[i]*x[i+1]
    return np.array(x)

# # %%
# d = [1 for i in range(shape)]
# x1 = np.array(thomas(A, d))
# x2 = np.linalg.solve(A, d)
# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2).mean())
# print(rmse(x1, x2))



