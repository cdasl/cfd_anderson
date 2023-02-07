import numpy as np
import matplotlib.pyplot as plt
from numpy import where
from thomas import tridiag, thomas

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

N = 21 # number of points
D = 1 # height of the pipe
dy = D/(N-1)

E = 1 # E=Δt/(Re_D*(Δy)^2)
Re_D = 5000 # 雷诺数
dt = E*Re_D*(dy)**2

u = np.zeros(N)
u[-1] = 1

def CN(E, u):
    """
    crank-nicolson method
    """
    A = -E/2
    B = 1+E
    K = (1-E)*u[1:-1]+E/2*(u[2:]+u[:-2])
    A_left = np.ones(N-3)*A
    B_left = np.ones(N-2)*B
    ABA = tridiag(A_left, B_left, A_left)
    K_right = K[:]
    K_right[-1] = K[-1]-A
    u_new = thomas(ABA, K_right)
    return u_new

if __name__=="__main__":
    t = 0
    while True:
        u_new = CN(E, u)
        t += dt
        if rmse(u_new, u[1:-1]) < 1e-7:
            break
        else:
            u[1:-1] = u_new
        if (t//dt) in [12, 36, 48, 250]:
            plt.plot(u[1:-1], label=str(t))
    print(u, t, t/dt)
    plt.legend(loc=2)
    plt.savefig("couette_cn.png")



    


