import numpy as np
import matplotlib.pyplot as plt
from numpy import where

N = 31 # number of points
DOMAIN_SIZE = 3 # length of the pipe
dx = DOMAIN_SIZE/(N-1)
x = np.linspace(0, DOMAIN_SIZE, N)

rho = 1.0-0.023*x # density
T = 1.0-0.009333*x # temperature
V = 0.05+0.11*x # velocity
gamma = 1.4 # specific heat capacity （比热比）
C = 0.5 # courant number, need c<=1 for stability


def Area_of_x(x):
    result = np.zeros_like(x)
    result[np.where(x<=1.5)] = 1+2.2*(x[np.where(x<=1.5)]-1.5)**2
    result[np.where(x>1.5)] = 1+0.2223*(x[np.where(x>1.5)]-1.5)**2
    return result

def get_radius_from_area(A):
    return np.sqrt(A/np.pi)

plt.plot(x, get_radius_from_area(Area_of_x(x)), x, -1*get_radius_from_area(Area_of_x(x)))
plt.savefig("pipe_2.png")



A = Area_of_x(x)

def Getdt(V, T):
    a = np.sqrt(T) # a is local sound speed (当地声速)
    return np.min(C*dx/(V+a))

# boundary condition
pn = 0.93
rho[-1] = 2*rho[-2]-rho[-3]
T[-1] = pn/rho[-1]
V[-1] = 2*V[-2]-V[-3]

# macCormack for de laval nozzle quation
def macCormack(rho, V, T):
    # rhs use forward diff
    rho_rhs = -(V[:-1]*(rho[1:]-rho[:-1])+rho[:-1]*(V[1:]-V[:-1])+rho[:-1]*V[:-1]*(np.log(A[1:])-np.log(A[:-1])))/dx
    V_rhs = (-V[:-1]*(V[1:]-V[:-1])-(1/gamma)*(T[1:]-T[:-1]+(T[:-1]/rho[:-1])*(rho[1:]-rho[:-1])))/dx
    T_rhs = (-V[:-1]*(T[1:]-T[:-1])-(gamma-1)*T[:-1]*(V[1:]-V[:-1]+V[:-1]*(np.log(A[1:])-np.log(A[:-1]))))/dx

    rho_bar = rho.copy()
    V_bar = V.copy()
    T_bar = T.copy()

    dt = Getdt(V,T)

    rho_bar[:-1] = rho_rhs*dt+rho[:-1]
    V_bar[:-1] = V_rhs*dt+V[:-1]
    T_bar[:-1] = T_rhs*dt+T[:-1]

    # rhs_bar use backward diff
    rho_rhs_bar = -(V_bar[1:]*(rho_bar[1:]-rho_bar[:-1])+rho_bar[1:]*(V_bar[1:]-V_bar[:-1])+rho_bar[1:]*V_bar[1:]*(np.log(A[1:])-np.log(A[:-1])))/dx
    V_rhs_bar = (-V_bar[1:]*(V_bar[1:]-V_bar[:-1])-(1/gamma)*(T_bar[1:]-T_bar[:-1]+(T_bar[1:]/rho_bar[1:])*(rho_bar[1:]-rho_bar[:-1])))/dx
    T_rhs_bar = (-V_bar[1:]*(T_bar[1:]-T_bar[:-1])-(gamma-1)*T_bar[1:]*(V_bar[1:]-V_bar[:-1]+V_bar[1:]*(np.log(A[1:])-np.log(A[:-1]))))/dx

    rho[1:] = 0.5*(rho[1:]+rho_bar[1:]+rho_rhs_bar*dt)
    V[1:] = 0.5*(V[1:]+V_bar[1:]+V_rhs_bar*dt)
    T[1:] = 0.5*(T[1:]+T_bar[1:]+T_rhs_bar*dt)

    return rho[1:-1], V[1:-1], T[1:-1]



if __name__ == "__main__":

    rho_plot = []
    V_plot = []
    T_plot = []

    for i in range(5000):
        rho[1:-1], V[1:-1], T[1:-1] = macCormack(rho, V, T)

        rho[0] = 1
        T[0] = 1
        V[0] = 2*V[1]-V[2]

        rho[-1] = 2*rho[-2]-rho[-3]
        T[-1] = pn/rho[-1]
        V[-1] = 2*V[-2]-V[-3]

        rho_plot.append(rho[15])
        V_plot.append(V[15])
        T_plot.append(T[15])


    print(rho[30], V[30], T[30])    
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
    fig.savefig("rho_v_t_ma_2.png", dpi=200)

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

    axs[4].plot(rho*T)
    axs[4].set_xlabel('x')
    axs[4].set_ylabel('Pressure')

    fig.tight_layout()
    fig.savefig("rho_v_t_ma_in_x_2.png", dpi=200)




