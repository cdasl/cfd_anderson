import numpy as np
import matplotlib.pyplot as plt
from numpy import where
np.set_printoptions(precision=3)

N = 61 # number of points
DOMAIN_SIZE = 3 # length of the pipe
dx = DOMAIN_SIZE/(N-1)
x = np.linspace(0, DOMAIN_SIZE, N)

def Area_of_x(x):
    return 1+2.2*(x-1.5)**2

# def dlnA_dx(x):
#     return 4.4*(x-1.5)/(Area_of_x(x))

A = Area_of_x(x)
# dlnAdx = dlnA_dx(x)

rho = np.zeros_like(x) # density
rho[np.where(x<=0.5)] = 1.0
rho[np.where(np.logical_and(0.5<=x , x<=1.5))] = 1.0-0.366*(x[np.where(np.logical_and(0.5<=x , x<=1.5))]-0.5)
rho[np.where(np.logical_and(1.5<=x , x<=2.1))] = 0.634-0.702*(x[np.where(np.logical_and(1.5<=x , x<=2.1))]-1.5)
rho[np.where(np.logical_and(2.1<=x , x<=3.0))] = 0.5892+0.10228*(x[np.where(np.logical_and(2.1<=x , x<=3.0))]-2.1)

T = np.zeros_like(x) # temperature
T[np.where(x<=0.5)] = 1.0
T[np.where(np.logical_and(0.5<=x , x<=1.5))] = 1.0-0.167*(x[np.where(np.logical_and(0.5<=x , x<=1.5))]-0.5)
T[np.where(np.logical_and(1.5<=x , x<=2.1))] = 0.833-0.4908*(x[np.where(np.logical_and(1.5<=x , x<=2.1))]-1.5)
T[np.where(np.logical_and(2.1<=x , x<=3.0))] = 0.93968+0.0622*(x[np.where(np.logical_and(2.1<=x , x<=3.0))]-2.1)

V = 0.59/(rho*A) # velocity
p = rho*T # Pressure

gamma = 1.4 # specific heat capacity （比热比）
C = 0.5 # courant number, need c<=1 for stability
Cx = 0.2 # use for artifical viscosity 

def Getdt(V, T):
    a = np.sqrt(T) # a is local sound speed (当地声速)
    return np.min(C*dx/(abs(V)+a))

U1 = rho*A
U2 = rho*A*V
U3 = rho*(T/(gamma-1)+0.5*gamma*V**2)*A

# macCormack for de laval nozzle quation
def macCormack(U1, U2, U3, dt):

    F1 = U2
    F2 = U2**2/U1+(gamma-1)/gamma*(U3-0.5*gamma*U2**2/U1)
    F3 = gamma*U2*U3/U1-gamma*(gamma-1)*0.5*U2**3/U1**2

    # rhs use forward diff
    U1_rhs = -(F1[2:]-F1[1:-1])/dx
    U2_rhs = -(F2[2:]-F2[1:-1])/dx+(gamma-1)/gamma*(U3[1:-1]-0.5*gamma*U2[1:-1]**2/U1[1:-1])*(np.log(A[2:])-np.log(A[1:-1]))/dx
    U3_rhs = -(F3[2:]-F3[1:-1])/dx

    U1_bar = U1.copy()
    U2_bar = U2.copy()
    U3_bar = U3.copy()

    # BUG2: forget abs() 
    U1_bar[1:-1] = U1_rhs*dt+U1[1:-1]+(Cx*(U1[2:]+U1[:-2]-2*U1[1:-1])*abs(p[2:]+p[:-2]-2*p[1:-1])/(p[2:]+p[:-2]+2*p[1:-1]))
    U2_bar[1:-1] = U2_rhs*dt+U2[1:-1]+(Cx*(U2[2:]+U2[:-2]-2*U2[1:-1])*abs(p[2:]+p[:-2]-2*p[1:-1])/(p[2:]+p[:-2]+2*p[1:-1]))
    U3_bar[1:-1] = U3_rhs*dt+U3[1:-1]+(Cx*(U3[2:]+U3[:-2]-2*U3[1:-1])*abs(p[2:]+p[:-2]-2*p[1:-1])/(p[2:]+p[:-2]+2*p[1:-1]))

    V_bar = U2_bar/U1_bar
    p_bar = U1_bar/A*(gamma-1)*(U3_bar/U1_bar-0.5*gamma*V_bar**2)

    F1_bar = U2_bar
    F2_bar = U2_bar**2/U1_bar+(gamma-1)/gamma*(U3_bar-0.5*gamma*U2_bar**2/U1_bar)
    F3_bar = gamma*U2_bar*U3_bar/U1_bar-gamma*(gamma-1)*0.5*U2_bar**3/U1_bar**2

    # rhs_bar use backward diff
    U1_rhs_bar = -(F1_bar[1:-1]-F1_bar[:-2])/dx
    U2_rhs_bar = -(F2_bar[1:-1]-F2_bar[:-2])/dx+(gamma-1)/gamma*(U3_bar[1:-1]-0.5*gamma*U2_bar[1:-1]**2/U1_bar[1:-1])*(np.log(A[1:-1])-np.log(A[:-2]))/dx
    U3_rhs_bar = -(F3_bar[1:-1]-F3_bar[:-2])/dx

    U1[1:-1] = 0.5*(U1[1:-1]+U1_bar[1:-1]+U1_rhs_bar*dt) + \
        (Cx*(U1_bar[2:]+U1_bar[:-2]-2*U1_bar[1:-1])*abs(p_bar[2:]+p_bar[:-2]-2*p_bar[1:-1])/(p_bar[2:]+p_bar[:-2]+2*p_bar[1:-1]))
    
    U2[1:-1] = 0.5*(U2[1:-1]+U2_bar[1:-1]+U2_rhs_bar*dt) + \
        (Cx*(U2_bar[2:]+U2_bar[:-2]-2*U2_bar[1:-1])*abs(p_bar[2:]+p_bar[:-2]-2*p_bar[1:-1])/(p_bar[2:]+p_bar[:-2]+2*p_bar[1:-1]))
    
    U3[1:-1] = 0.5*(U3[1:-1]+U3_bar[1:-1]+U3_rhs_bar*dt) + \
        (Cx*(U3_bar[2:]+U3_bar[:-2]-2*U3_bar[1:-1])*abs(p_bar[2:]+p_bar[:-2]-2*p_bar[1:-1])/(p_bar[2:]+p_bar[:-2]+2*p_bar[1:-1]))

    return U1[1:-1], U2[1:-1], U3[1:-1]



if __name__ == "__main__":

    rho_plot = []
    V_plot = []
    T_plot = []

    for i in range(1400):
        dt = Getdt(V,T)
        U1[1:-1], U2[1:-1], U3[1:-1] = macCormack(U1, U2, U3, dt)

        rho[0] = 1
        T[0] = 1
        U1[0] = rho[0]*A[0]
        U2[0] = 2*U2[1]-U2[2]
        V[0] = U2[0]/U1[0]
        U3[0] = rho[0]*(T[0]/(gamma-1)+0.5*gamma*V[0]**2)*A[0]

        U1[-1] = 2*U1[-2]-U1[-3]
        U2[-1] = 2*U2[-2]-U2[-3]
        # BUG1: forget to update V[-1]
        V[-1] = U2[-1]/U1[-1]  
        U3[-1] = 0.6784*A[-1]/(gamma-1)+0.5*gamma*U2[-1]*V[-1]

        rho = U1/A
        V = U2/U1
        T = (gamma-1)*(U3/U1-0.5*gamma*V**2)
        p = rho*T

        rho_plot.append(rho[15])
        V_plot.append(V[15])
        T_plot.append(T[15])


    print(rho[15], V[15], T[15])    
    fig, axs = plt.subplots(4, 1, figsize=(10,10))
    axs[0].plot(rho_plot)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Density')

    axs[1].plot(V_plot)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Velocity')

    axs[2].plot(T_plot)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Temperature')

    axs[3].plot(V_plot/np.sqrt(T_plot))
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Ma')

    fig.tight_layout()
    fig.savefig("rho_v_t_ma_4.png", dpi=200)

    fig, axs = plt.subplots(5, 1, figsize=(10,10))
    axs[0].plot(rho)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Density')

    axs[1].plot(V)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Velocity')

    axs[2].plot(T)
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Temperature')

    axs[3].plot(V/np.sqrt(T))
    axs[3].set_xlabel('x')
    axs[3].set_ylabel('Ma')

    axs[4].plot(rho*V*A)
    axs[4].set_xlabel('x')
    axs[4].set_ylabel('mass flux')

    fig.tight_layout()
    fig.savefig("rho_v_t_ma_in_x_4.png", dpi=200)




