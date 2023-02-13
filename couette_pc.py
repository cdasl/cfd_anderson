import numpy as np
import matplotlib.pyplot as plt
from numpy import where

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

NX = 21 # number of points in x-axis
NY = 11 # number of points in y-axis
width = 0.5 
height = 0.01
rho = 0.002377 # density
Re = 63.6 # 雷诺数
ue = 1 # initial velocity
mu = rho*ue*height/Re
alpha = 0.1 # relaxation factor

p_star = np.zeros((NX, NY)) # pressure
p_prime = np.zeros((NX, NY)) # pressure correction

u = np.zeros((NX+1, NY)) # x velocity
v = np.zeros((NX+2, NY+1)) # y velocity

dx = width/(NX-1)
dy = height/(NY-1)
dt = 1e-3

# boundary condition
u[:,-1] = ue

# Initial contidion: add velocity pulse
v[15-1, 5-1] = 0.5


v15_5_plot = []

def PC(rho, u, v, p_star, p_prime):
    """
    pressure correction method
    """
    # step 1: set p_star value in internal grid
    # step 2: solve rho*u_star and rho*v_star in next step by momentum equation
    # step 3: use the predicted values to solve p_prime for using the pressure correction formula.
    u_star = u.copy() # (nx+1,ny)
    v_star = v.copy() # (nx+2,ny+1)

    rho_u_star = rho*u_star
    rho_v_star = rho*v_star

    u0 = u.shape[0]-1
    u1 = u.shape[1]-1
    v_bar = 0.5*(v[1:u0,2:u1+1]+v[2:u0+1,2:u1+1])
    v_bar_bar = 0.5*(v[1:u0,1:u1]+v[2:u0+1,1:u1]) # (nx, ny-1)

    tmp1 = ((rho*u*u)[2:,1:-1]-(rho*u*u)[:-2,1:-1])/(2*dx)
    tmp2 = (rho*u[1:-1,2:]*v_bar-rho*v_bar_bar)/(2*dy)
    tmp3 = (u[2:,1:-1]+u[:-2,1:-1]-2*u[1:-1,1:-1])/(dx**2)
    tmp4 = (u[1:-1,2:]+u[1:-1,:-2]-2*u[1:-1,1:-1])/(dy**2)
    A_star = -(tmp1+tmp2)+mu*(tmp3+tmp4)
    rho_u_star[1:-1,1:-1] = rho*u_star[1:-1,1:-1] + A_star*dt-dt/dx*(p_star[1:u0,1:u1]-p_star[:u0-1,1:u1])


    v0 = v.shape[0]-1
    v1 = v.shape[1]-1
    u_bar = 0.5*(u[1:v0,:v1-1]+u[1:v0,1:v1])
    u_bar_bar = 0.5*(u[:v0-1,:v1-1]+u[:v0-1,1:v1])
    
    tmp5 = (rho*v[2:,1:-1]*u_bar-rho*v[:-2,1:-1]*u_bar_bar)/(2*dx)
    tmp6 = ((rho*v*v)[1:-1,2:]-(rho*v*v)[1:-1,:-2])/(2*dy)
    tmp7 = (v[2:,1:-1]+v[:-2,1:-1]-2*v[1:-1,1:-1])/(dx**2)
    tmp8 = (v[1:-1,2:]+v[1:-1,:-2]-2*v[1:-1,1:-1])/(dy**2)
    B_star = -(tmp5+tmp6)+mu*(tmp7+tmp8)
    rho_v_star[1:-1,1:-1] = rho*v_star[1:-1,1:-1] + B_star*dt-dt/dy*(p_star[:v0-1,1:v1]-p_star[:v0-1,:v1-1])

    # boundary condition for rho_u_star and rho_v_star
    rho_u_star[:,-1] = rho*ue # top
    rho_u_star[:,0] = 0 # bottom
    rho_u_star[0,:] = rho_u_star[1,:] # left
    rho_u_star[-1,:] = rho_u_star[-2,:] # right

    rho_v_star[0,:] = 0 # left
    rho_v_star[-1,:] = rho_v_star[-2,:] # right
    rho_v_star[:,0] = 0 # bottom
    rho_v_star[:,-1] = 0 # top

    a = 2*(dt/dx**2+dt/dy**2)
    b = -dt/dx**2
    c = -dt/dy**2

    p0 = p_star.shape[0]-1
    p1 = p_star.shape[1]-1
    while True:
        d = 1/dx*(rho_u_star[2:p0+1,1:p1]-rho_u_star[1:p0,1:p1]) + \
            1/dy*(rho_v_star[2:p0+1,2:p1+1]-rho_v_star[2:p0+1,1:p1])
        p_prime_new = -1/a*(b*p_prime[2:,1:-1]+b*p_prime[:-2,1:-1]+c*p_prime[1:-1,2:]+c*p_prime[1:-1,:-2]+d)
        diff = rmse(p_prime[1:-1,1:-1], p_prime_new)
        # if np.sum(d) < 1e-10:
        print("sum(d)=",np.sum(d))
        if diff < 1e-7:
            p_prime[1:-1,1:-1] = p_prime_new
            break
        else:
            p_prime[1:-1,1:-1] = p_prime_new
    
    p = p_star + alpha*p_prime
    u = rho_u_star/rho
    v = rho_v_star/rho

    v15_5_plot.append(v[15-1,5-1])

    return p, u, v, p_prime
    





if __name__=="__main__":
    for i in range(500):
        p, u, v, p_prime = PC(rho, u, v, p_star, p_prime)
        p_star = p
    
    plt.plot(v15_5_plot)
    plt.title('y velocity at grid point(15,5)')
    plt.xlabel('Iterations')
    plt.ylabel('y veloctiy ft/s')
    plt.xlim([-10,500])
    plt.ylim([-0.06,0.35])
    plt.savefig("v15_5_plot.png")

    xx = np.linspace(0, width, NX)
    yy = np.linspace(0, height, NY)
    X, Y = np.meshgrid(xx, yy)

    levels = 24 # number of contours
    f = plt.figure(figsize=(18,9))
    plt.subplot(2, 1, 1,)
    plt.contourf(np.transpose(u),levels, cmap="coolwarm")
    plt.colorbar(label='ft/s')
    plt.title('u Vel')
    plt.subplot(2,1,2)
    plt.contourf(np.transpose(v),levels, cmap="coolwarm")
    plt.title('v Vel')
    plt.colorbar(label='ft/s')
    plt.savefig("couette_pc_velocity.png")

    f = plt.figure(figsize=(18,4.5))
    x = np.arange(0,p.shape[0], 1)
    y = np.arange(0,p.shape[1], 1)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X,Y,np.transpose(p),levels)
    plt.colorbar()
    plt.title('Pressure')
    plt.savefig("couette_pc_pressure.png")






    


