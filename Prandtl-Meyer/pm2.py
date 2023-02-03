import numpy as np
import matplotlib.pyplot as plt
from numpy import where

N = 81 # number of points
width = 65 # width of the pipe
height_fixed = 40 # fixed height of the pipe
theta_deg = 5.352 # degree of theta
theta_rad = theta_deg*np.pi/180 # rad of theta
E = 10 # position of angle

gamma = 1.4 # specific heat capacity （比热比）
Cy = 1.0 # need for artifical viscosity
Courant = 0.5 # courant number, need c<=1 for stability

sound_speed = 339
u = np.ones(N)*2*sound_speed # velocity of x
v = np.zeros(N) # velocity of y
p = np.ones(N)*1.01e5 # pressure
rho = np.ones(N)*1.23 # density
T = np.ones(N)*286.1 # temperature
R = 287.01 # gas constant（气体常数） R=p/(rho*T)
Ma = np.ones(N)*2 # mach number

# dx = width/(N-1)
# x = np.linspace(0, width, N)

# h = np.zeros(N) # 上下壁面的距离
# ys = np.zeros(N) # 下壁面的纵坐标

# pivot = 0

# for idx, val in range(width):
#     if 0 <= val <= E:
#         h[idx] = 40
#         ys[idx] = 0
#     elif E <= val <=65:
#         h[idx] = 40+(val-10)*np.tan(theta_rad)
#         ys[idx] = -(val-10)*np.tan(theta_rad)

# max_eta = 1
# eta = np.linspace(0, max_eta, N)
# d_eta = max_eta/(N-1)
# y = h*eta+ys

# d_ksi_d_x = np.ones(N)
# d_ksi_d_y = np.zeros(N)
# d_eta_d_x = np.zeros(N)
# for idx, val in enumerate(x):
#     if 0 <= val <= E:
#         d_eta_d_x[idx] = 0
#     elif E <= val <=65:
#         d_eta_d_x[idx] = (1-eta[idx])*np.tan(theta_rad)/h[idx]
# d_eta_d_y = 1/h

def calc_F(rho, u, v, p):
    F1 = rho*u
    F2 = F1*u+p
    F3 = F1*v
    F4 = gamma/(gamma-1)*p*u+F1*(u**2+v**2)*0.5
    F = np.vstack((F1, F2, F3, F4))
    return F

def calc_ABC(F):
    F1 = F[0]
    F2 = F[1]
    F3 = F[2]
    F4 = F[3]
    A = F3**2/(2*F1)-F4
    B = gamma/(gamma-1)*F1*F2
    C = -0.5*(gamma+1)/(gamma-1)*F1**3
    ABC = np.vstack((A, B, C))
    return ABC

def calc_rho_u_v_p_T(F):
    A, B, C = calc_ABC(F)

    rho = (-B+np.sqrt(B**2-4*A*C))/(2*A)
    u = F[0]/rho
    v = F[2]/F[0]
    p = F[1]-F[0]*u
    T = p/(rho*R)

    return rho, u, v, p, T


def calc_G(F):
    ABC = calc_ABC(F)
    A = ABC[0]
    B = ABC[1]
    C = ABC[2]
    F1 = F[0]
    F2 = F[1]
    F3 = F[2]
    F4 = F[3]

    rho = (-B+np.sqrt(B**2-4*A*C))/(2*A)
    u = F1/rho
    p = F2-F1*u
    G1 = rho*F3/F1
    G2 = F3
    G3 = rho*(F3/F1)**2+p
    G4 = gamma/(gamma-1)*(F2-F1**2/rho)*F3/F1+0.5*rho*F3*((F1/rho)**2+(F3/F1)**2)/F1
    G = np.vstack((G1, G2, G3, G4))
    return G

def backOutMach(f,gamma):
    # given some value of f, we need to backout M
    # f input in deg converted to rad
    guessM = 4
    delta = 1
    while abs(delta)>1e-7:
        fGuess = np.sqrt((gamma+1)/(gamma-1)) * np.arctan(np.sqrt( ( (gamma-1)/(gamma+1) ) * (guessM**2 -1) )) - np.arctan(np.sqrt(guessM**2 -1)) 
        delta = fGuess - f
        guessM = guessM - delta*0.1
    return guessM

def mach2Primitives(Mcal,Mact,pcal,Tcal):
    # Calculate primitives from mach numbers and new p & T
    top = 1 + ((gamma-1)/2)*Mcal**2
    bottom = 1 + ((gamma-1)/2)*Mact**2
    pAct = pcal * (top/bottom) ** ( gamma/(gamma-1) )
    TAct = Tcal * (top/bottom)
    rhoAact = pAct/(R*TAct)
    return pAct,TAct,rhoAact

def wallBC(rho, u, v, p, x):
    if x <= 10:
        phi = np.arctan(v/u)
        theta = 0
    else:
        phi = theta_rad-np.arctan(abs(v)/u)
        theta = theta_rad
    T = p/(rho*R)
    a =((gamma*p/rho)**0.5)
    Ma_cal = np.sqrt(u**2+v**2)/a
    f_cal = (np.sqrt((gamma+1)/(gamma-1))) * (np.arctan(np.sqrt((gamma-1)/(gamma+1)*(Ma_cal**2-1)))) - np.arctan(np.sqrt(Ma_cal**2-1))
    f_act = f_cal + phi
    Ma_act = backOutMach(f_act, gamma)
    pAct,TAct,rhoAct = mach2Primitives(Ma_cal,Ma_act,p,T)
    vNew = -u*np.tan(theta)# v is the tan(theta) component of u
    uNew = u # u is kept the same 
    return rhoAct, uNew, vNew, pAct

yValues = [] # for drawing figure

def macCormack(rho, u, v, p, T, Ma, x):
    if x <= 10:
        h = height_fixed # 上下壁面的距离
        ys = 0 # 下壁面的坐标
        d_eta_d_x = np.zeros(N)
    elif 10 <= x <= 65:
        h = 40+(x-10)*np.tan(theta_rad)
        ys = -(x-10)*np.tan(theta_rad)
        eta = np.linspace(0, 1, N)
        d_eta_d_x = (1-eta)*np.tan(theta_rad)/h

    yValues.append(np.linspace(ys, height_fixed, N))
    
    # calc delta_x and delta_ksi
    d_eta = 1/(N-1)
    dy = h/(N-1)
    mu = np.arcsin(1/Ma)
    u_v_angle = np.arctan(v/u)
    dx_plus = np.min(dy/abs(np.tan(u_v_angle+mu)))
    dx_minus = np.min(dy/abs(np.tan(u_v_angle-mu)))
    dx = Courant*min(dx_plus, dx_minus)
    d_ksi = dx


    F = calc_F(rho, u, v, p)
    G = calc_G(F)

    # forward diff
    F_rhs = np.zeros((F.shape))
    for i in range(4):
        F_rhs[i][:-1] = d_eta_d_x[:-1]*(F[i][:-1]-F[i][1:])/d_eta+1/h*(G[i][:-1]-G[i][1:])/d_eta
    
    F_bar = np.zeros((F.shape))

    SF = np.zeros((F.shape))
    for i in range(4):
        SF[i][1:-1] = Cy*abs(p[2:]+p[:-2]-2*p[1:-1])*(F[i][2:]+F[i][:-2]-2*F[i][1:-1])/(p[2:]+p[:-2]+2*p[1:-1])

    for i in range(4):
        F_bar[i][:-1] = F_rhs[i][:-1]*d_ksi+F[i][:-1]+SF[i][:-1]
        F_bar[i][-1] = F[i][-1]
    
    G_bar = calc_G(F_bar)
    _,__, ___, p_bar, _____ = calc_rho_u_v_p_T(F_bar)
    p_bar[-1] = p[-1]


    SF_bar = np.zeros(F.shape)
    for i in range(4):
        SF_bar[i][1:-1] = Cy*abs(p_bar[2:]+p_bar[:-2]-2*p_bar[1:-1])* \
            (F_bar[i][2:]+F_bar[i][:-2]-2*F_bar[i][1:-1]) \
            /(p_bar[2:]+p_bar[:-2]+2*p_bar[1:-1])
    
    # backward diff
    F_rhs_bar = np.zeros((F.shape))
    for i in range(4):
        F_rhs_bar[i][1:] = d_eta_d_x[1:]*(F_bar[i][:-1]-F_bar[i][1:])/d_eta+1/h*(G_bar[i][:-1]-G_bar[i][1:])/d_eta
        F_rhs_bar[i][0] = d_eta_d_x[0]*(F_bar[i][0]-F_bar[i][1])/d_eta+1/h*(G_bar[i][0]-G_bar[i][1])/d_eta
    
    for i in range(4):
        F[i][1:] = 0.5*(F[i][1:]+F_bar[i][1:]+F_rhs_bar[i][1:]*d_ksi)+SF_bar[i][1:]
        F[i][0] = 2*F[i][1]-F[i][2]
    
    rho, u, v, p, T = calc_rho_u_v_p_T(F)
    rho[0], u[0], v[0], p[0] = wallBC(rho[0], u[0], v[0], p[0], x)
    Ma = ((u**2 + v**2)**0.5)/((gamma*p/rho)**0.5)

    new_x = x+dx

    return rho, u, v, p, T, Ma, new_x

    
if __name__=="__main__":
    rho_plot = []
    u_plot = []
    v_plot = []
    p_plot = []
    T_plot = []
    xValues = []

    x = 0
    while x <= width:
        xValues.append(x)
        rho, u, v, p, T, Ma, new_x = macCormack(rho, u, v, p, T, Ma, x)
        x = new_x
        rho_plot.append(rho[:])
        u_plot.append(u[:])
        v_plot.append(v[:])
        p_plot.append(p[:])
        T_plot.append(T[:])
    
    u_plot = np.array(u_plot).T
    v_plot = np.array(v_plot).T
    p_plot = np.array(p_plot).T
    T_plot = np.array(T_plot).T
    rho_plot = np.array(rho_plot).T

    xValues=np.array(xValues)
    print(xValues)
    yValues=np.transpose(np.array(yValues))

    print(p_plot.shape)



    xx = np.linspace(0,65,200)
    hh = np.zeros(200)
    index = 0
    for i in xx:
        if i <10:
            hh[index] = 40
        else:
            hh[index] = 40+np.tan(0.09341002)*(xx[index]-10)
        index = index +1
    #plot geometry
    plt.figure(figsize=(16,12))
    plt.plot(xx,np.ones(200)*40,'black')
    plt.plot(xx,40-hh,'black')
    plt.ylim([-15,50])
    plt.xlim([0,65])
    plt.grid()
    plt.title('2D Supersonic Flow: Ramp Angle = 5.352 deg')
    plt.xlabel('x')
    plt.ylabel('y')

    # Plot Results
    xMatrix = np.array([xValues,]*N)
    plt.contourf(xMatrix, yValues, p_plot, alpha=0.7)  
    cb = plt.colorbar(label='Pressure N/m^2')
    plt.quiver(xMatrix, yValues, u_plot, v_plot)
    plt.savefig("p.png")

    plt.contourf(xMatrix, yValues, T_plot, alpha=0.7)
    cb.remove()
    cb = plt.colorbar(label='Temperature K')
    # plt.quiver(xMatrix, yValues, u_plot, v_plot)
    plt.savefig("T.png")

    plt.contourf(xMatrix, yValues, rho_plot, alpha=0.7)
    cb.remove()
    cb = plt.colorbar(label='Density kg/m^3')
    # plt.quiver(xMatrix, yValues, u_plot, v_plot)
    plt.savefig("rho.png")

    plt.contourf(xMatrix, yValues, ((u_plot**2 + v_plot**2)**0.5)/((gamma*p_plot/rho_plot)**0.5), alpha=0.7)
    cb.remove()
    cb = plt.colorbar(label='Ma')
    # plt.quiver(xMatrix, yValues, u_plot, v_plot)
    plt.savefig("Ma.png")
    


    
    

    




    





