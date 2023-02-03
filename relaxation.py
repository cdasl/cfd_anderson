import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

NX = 101
NY = 101
DOMAIN_SIZE_X = 2
DOMAIN_SIZE_Y = 2
dx = DOMAIN_SIZE_X/(NX-1)
dy = DOMAIN_SIZE_Y/(NY-1)



Mat = np.zeros((NX, NY))
print(Mat.shape)
Mat[:, 0] = 0
Mat[:, -1] = 0
Mat[0, :] = 50
Mat[-1, :] = 0

sol = [Mat]
count = 0

def relaxation(A):
    count = 0
    omega = 0.5
    while True:
        Anew = A.copy()
        Anew[1:-1, 1:-1] = ((dy**2)*(A[2:, 1:-1]+A[:-2,1:-1])+(dx**2)*(A[1:-1, 2:]+A[1:-1,:-2]))/(2*(dx**2+dy**2))
        print(Anew)
        count += 1
        if rmse(A, Anew) < 1e-5:
            print("converged!")
            sol.append(Anew.copy())
            return Anew
        else:
            sol.append(Anew.copy())
            A = A+0.95*(Anew-A)



def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature at t = {k*1:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=50)
    plt.colorbar()

    return plt

# Do the calculation here
relaxation(Mat)
print(len(sol))

def animate(k):
    if k%100 == 0:
        plotheatmap(sol[k], k)

anim = animation.FuncAnimation(plt.figure(), animate, frames=len(sol),interval=1, repeat=False)
anim.save("relaxation.gif")

print("Done!")

