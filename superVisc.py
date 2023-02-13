import numpy as np
import matplotlib.pyplot as plt
import pdb
import h5py
################
#mu Calculation#
################
def calcMu(muRef,temp,tempRef):
    """
    Calculate local dynamic viscosity
    using Sutherland's Law
    """
    mu = muRef* ( (temp/tempRef)**(3/2) ) * ( (tempRef+110)/(temp+110) )
    return mu
############################
#Convert to U, F, E Vectors#
############################
def getUfromPrim(presDomain,tempDomain,uVelDomain,vVelDomain,muDomain,R,cv):
    U1 = presDomain/(R*tempDomain)
    U2 = U1 * uVelDomain
    U3 = U1 * vVelDomain
    U5 = U1 * (cv*tempDomain + (uVelDomain**2 + vVelDomain**2) / 2 )
    return U1,U2,U3,U5
def U2Prim(U1Bar,U2Bar,U3Bar,U5Bar,R,cv):
    rho = U1Bar
    uVelDomain = U2Bar/U1Bar
    vVelDomain = U3Bar/U1Bar
    e =   (U5Bar/U1Bar) - ( (uVelDomain**2) + (vVelDomain**2) )/2
    tempDomain = e/cv
    presDomain = U1Bar*R*tempDomain
    return presDomain,tempDomain,uVelDomain,vVelDomain
def getE(tauxyE,tauxyF,tauxxE,tauyyF,presDomain,tempDomain,uVelDomain,vVelDomain,qxE,R,cv):
    rho = presDomain/(R*tempDomain)
    E1 = uVelDomain*rho
    E2 = rho * uVelDomain**2 + presDomain - tauxxE
    E3 = rho * uVelDomain * vVelDomain - tauxyE
    Et = rho*(cv*tempDomain + (uVelDomain**2 + vVelDomain**2) /2)
    E5 = (Et + presDomain) * uVelDomain - uVelDomain*tauxxE - vVelDomain*tauxyE + qxE
    return E1,E2,E3,E5
def getF(tauxyE,tauxyF,tauxxE,tauyyF,presDomain,tempDomain,uVelDomain,vVelDomain,qyF,R,cv):
    rho = presDomain/(R*tempDomain)
    F1 = vVelDomain*rho
    F2 = rho * uVelDomain* vVelDomain -tauxyF
    F3 = rho * vVelDomain**2 + presDomain - tauyyF
    Et = rho*(cv*tempDomain + (uVelDomain**2 + vVelDomain**2)/2)
    F5 = (Et + presDomain) * vVelDomain - uVelDomain*tauxyF - vVelDomain*tauyyF + qyF
    return F1,F2,F3,F5
#########################
# Tau and q calculations#
#########################
def getTauxyfromPrim(uVelDomain,vVelDomain,muDomain,case,deltay,deltax):
    xMax,yMax = uVelDomain.shape
    tauxy = np.zeros(uVelDomain.shape)
    # Case 1: predictor for E
    if case ==1:
        for i in range(0,xMax):
            for j in range(0,yMax):
                # Calculate dudy
                if j==0:# Wall
                    dudy = (uVelDomain[i,j+1]-uVelDomain[i,j])/deltay
                elif j == yMax-1: # top
                    dudy = (uVelDomain[i,j]-uVelDomain[i,j-1])/deltay
                else:# j values between top and bottom
                    dudy = (uVelDomain[i,j+1]-uVelDomain[i,j-1])/(2*deltay)
                # Calculate dvdx
                if i==0: # left inlet
                    dvdx = (vVelDomain[i+1,j]-vVelDomain[i,j])/deltax
                else: # middle and right side
                    dvdx = (vVelDomain[i,j]-vVelDomain[i-1,j])/deltax
                tauxy[i,j] = muDomain[i,j]*(dudy + dvdx)
    # Case 2: predictor for F
    elif case == 2:
        for i in range(0,xMax):
            for j in range(0,yMax):
                # Calculate dudy
                if j ==0: # if we are at the wall then we cannot do rearward diff
                    dudy = (uVelDomain[i,j+1]-uVelDomain[i,j])/deltay
                else: #rearward diff
                    dudy = (uVelDomain[i,j] - uVelDomain[i,j-1])/deltay
                # Calculate dvdx
                if i ==0:
                    dvdx = (vVelDomain[i+1,j]-vVelDomain[i,j])/deltax
                elif i == xMax-1:
                    dvdx = (vVelDomain[i,j]-vVelDomain[i-1,j])/deltax
                else:
                    dvdx = (vVelDomain[i+1,j]-vVelDomain[i-1,j])/(2*deltax)
                tauxy[i,j] = muDomain[i,j]*(dudy + dvdx)
    # Case 3: corrector for E
    elif case == 3:
        for i in range(0,xMax):
            for j in range(0,yMax):
                # Calculate dudy
                if j==0:# Wall
                    dudy = (uVelDomain[i,j+1]-uVelDomain[i,j])/deltay
                elif j == yMax-1: # top
                    dudy = (uVelDomain[i,j]-uVelDomain[i,j-1])/deltay
                else:# j values between top and bottom
                    dudy = (uVelDomain[i,j+1]-uVelDomain[i,j-1])/(2*deltay)
                # Calculate dvdx
                if i==xMax-1: # outlet
                    dvdx = (vVelDomain[i,j]-vVelDomain[i-1,j])/deltax
                else: # middle and left side
                    dvdx = (vVelDomain[i+1,j]-vVelDomain[i,j])/deltax
                tauxy[i,j] = muDomain[i,j]*(dudy + dvdx)
    # Case 4: corrector for F
    elif case == 4:
        for i in range(0,xMax):
            for j in range(0,yMax):
                # Calculate dudy
                if j ==yMax-1: # if we are at the wall then we cannot do rearward diff
                    dudy = (uVelDomain[i,j]-uVelDomain[i,j-1])/deltay
                else: #rearward diff
                    dudy = (uVelDomain[i,j+1] - uVelDomain[i,j])/deltay
                # Calculate dvdx
                if i ==0:
                    dvdx = (vVelDomain[i+1,j]-vVelDomain[i,j])/deltax
                elif i == xMax-1:
                    dvdx = (vVelDomain[i,j]-vVelDomain[i-1,j])/deltax
                else:
                    dvdx = (vVelDomain[i+1,j]-vVelDomain[i-1,j])/(2*deltax)
                tauxy[i,j] = muDomain[i,j]*(dudy + dvdx)
    else:
        print('ERROR:No case selected. Must be 1, 2, 3, or 4')
    return tauxy
def getTauxx(uVelDomain,vVelDomain,muDomain,case,deltay,deltax):
    # Calculate Tauxx for predictor
    xMax,yMax = uVelDomain.shape
    tauxx = np.zeros(uVelDomain.shape)
    # Case 1: predictor for E
    if case ==1:
        for i in range(0,xMax):
            for j in range(0,yMax):
                # Calculate dvdy with central diff
                if j==0:# Wall
                    dvdy = (vVelDomain[i,j+1]-vVelDomain[i,j])/deltay
                elif j == yMax-1: # top
                    dvdy = (vVelDomain[i,j]-vVelDomain[i,j-1])/deltay
                else:# j values between top and bottom
                    dvdy = (vVelDomain[i,j+1]-vVelDomain[i,j-1])/(2*deltay)

                # Calculate dudx with rearward diff
                if i==0: # left inlet
                    dudx = (uVelDomain[i+1,j]-uVelDomain[i,j])/deltax
                else: # middle and right side
                    dudx = (uVelDomain[i,j]-uVelDomain[i-1,j])/deltax
                tauxx[i,j] = -(2/3) * muDomain[i,j] * (dudx + dvdy) + 2 * muDomain[i,j] * dudx
    # Case 2: corrector for E
    elif case == 2:
        for i in range(0,xMax):
            for j in range(0,yMax):
                # Calculate dvdy with central diff
                if j==0:# Wall
                    dvdy = (vVelDomain[i,j+1]-vVelDomain[i,j])/deltay
                elif j == yMax-1: # top
                    dvdy = (vVelDomain[i,j]-vVelDomain[i,j-1])/deltay
                else:# j values between top and bottom
                    dvdy = (vVelDomain[i,j+1]-vVelDomain[i,j-1])/(2*deltay)

                # Calculate dudx with forward diff
                if i==xMax-1: # right outlet
                    dudx = (uVelDomain[i,j]-uVelDomain[i-1,j])/deltax
                else: # middle and right side
                    dudx = (uVelDomain[i+1,j]-uVelDomain[i,j])/deltax
                tauxx[i,j] = -(2/3) * muDomain[i,j] * (dudx + dvdy) + 2 * muDomain[i,j] * dudx
    # Calculate Tauxx for corrector
    return tauxx
def getTauyy(uVelDomain,vVelDomain,muDomain,case,deltay,deltax):
    xMax,yMax = uVelDomain.shape
    tauyy = np.zeros(uVelDomain.shape)
    # Calculate tauyy for predictor
    if case == 1:
        for i in range(0,xMax):
            for j in range(0,yMax):
                # Calculate dvdy with rearward
                if j ==0: # if we are at the wall then we cannot do rearward diff
                    dvdy = (vVelDomain[i,j+1]-vVelDomain[i,j])/deltay
                else: #rearward diff
                    dvdy = (vVelDomain[i,j] - vVelDomain[i,j-1])/deltay

                # Calculate dudx with central diff
                if i ==0:
                    dudx = (uVelDomain[i+1,j]-uVelDomain[i,j])/deltax
                elif i == xMax-1:
                    dudx = (uVelDomain[i,j]-uVelDomain[i-1,j])/deltax
                else:
                    dudx = (uVelDomain[i+1,j]-uVelDomain[i-1,j])/(2*deltax)
                tauyy[i,j] = -(2/3) * muDomain[i,j] * (dudx + dvdy) + 2 * muDomain[i,j] * dvdy

    # Calculate tauyy for corrector
    elif case == 2:
        for i in range(0,xMax):
            for j in range(0,yMax):
                # Calculate dvdy with forward diff
                if j ==yMax-1: # outlet
                    dvdy = (vVelDomain[i,j]-vVelDomain[i,j-1])/deltay
                else: #forward diff
                    dvdy = (vVelDomain[i,j+1] - vVelDomain[i,j])/deltay

                # Calculate dudx with central diff
                if i ==0:
                    dudx = (uVelDomain[i+1,j]-uVelDomain[i,j])/deltax
                elif i == xMax-1:
                    dudx = (uVelDomain[i,j]-uVelDomain[i-1,j])/deltax
                else:
                    dudx = (uVelDomain[i+1,j]-uVelDomain[i-1,j])/(2*deltax)
                tauyy[i,j] = -(2/3) * muDomain[i,j] * (dudx + dvdy) + 2 * muDomain[i,j] * dvdy
    return tauyy
def getqxE(tempDomain,cp,muDomain,deltax,case):
    xMax,yMax = tempDomain.shape
    qxE = np.zeros(tempDomain.shape)
    # Case 1: predictor for E
    if case ==1:
        for i in range(0,xMax):
            for j in range(0,yMax):
                # Calculate dudx with rearward diff
                if i==0: # left inlet
                    dTdx = (tempDomain[i+1,j]-tempDomain[i,j])/deltax
                else: # middle and right side
                    dTdx = (tempDomain[i,j]-tempDomain[i-1,j])/deltax
                qxE[i,j] = -((muDomain[i,j]*cp)/0.71)*dTdx
    # Case 2: corrector for E
    elif case == 2:
        for i in range(0,xMax):
            for j in range(0,yMax):
                # Calculate dudx with forward diff
                if i==xMax-1: # right outlet
                    dTdx = (tempDomain[i,j]-tempDomain[i-1,j])/deltax
                else: # middle and right side
                    dTdx = (tempDomain[i+1,j]-tempDomain[i,j])/deltax
                qxE[i,j] = -((muDomain[i,j]*cp)/0.71)*dTdx
    return qxE
def getqyF(tempDomain,cp,muDomain,deltay,case):
    xMax,yMax = tempDomain.shape
    qyF = np.zeros(tempDomain.shape)
    # Predictor step
    if case == 1:
        for i in range(0,xMax):
            for j in range(0,yMax):
                # Calculate dvdy with rearward
                if j ==0: # if we are at the wall then we cannot do rearward diff
                    dTdy = (tempDomain[i,j+1]-tempDomain[i,j])/deltay
                else: #rearward diff
                    dTdy = (tempDomain[i,j] - tempDomain[i,j-1])/deltay
                qyF[i,j] = -((muDomain[i,j]*cp)/0.71)*dTdy
    # Corrector step calculation
    elif case == 2:
        for i in range(0,xMax):
            for j in range(0,yMax):
                # Calculate dvdy with forward diff
                if j ==yMax-1: # outlet
                    dTdy = (tempDomain[i,j]-tempDomain[i,j-1])/deltay
                else: #forward diff
                    dTdy = (tempDomain[i,j+1] - tempDomain[i,j])/deltay
                qyF[i,j] = -((muDomain[i,j]*cp)/0.71)*dTdy
    return qyF
#####################
#Boundary Conditions#
#####################
def initBoundaryCond(presDomain,tempDomain,uVelDomain,vVelDomain,tempWall):
    """
    Calculation of boundary conditions
    """
    # Case 1. [0,0] leading edge
    uVelDomain[0,0] = 0
    vVelDomain[0,0] = 0
    # Case 2. [0,1:] inflow and [:.-1] top
    vVelDomain[0,1:]  = 0  # not including leading edge
    vVelDomain[:,-1]  = 0
    # Case 3. [:,0] bottom (flat plate wall)
    presDomain[1:,0] = 2*presDomain[1:,1] - presDomain[1:,2]
    tempDomain[1:,0] = tempWall
    uVelDomain[1:,0] = 0
    vVelDomain[1:,0] = 0
    # Case 4. [-1,1:-1] Outflow
    presDomain[-1,1:-1] = 2*presDomain[-2,1:-1] - presDomain[-3,1:-1]
    tempDomain[-1,1:-1] = 2*tempDomain[-2,1:-1] - tempDomain[-3,1:-1]
    uVelDomain[-1,1:-1] = 2*uVelDomain[-2,1:-1] - uVelDomain[-3,1:-1]
    vVelDomain[-1,1:-1] = 2*vVelDomain[-2,1:-1] - vVelDomain[-3,1:-1]
    return presDomain,tempDomain,uVelDomain,vVelDomain
def updateBoundaryCond(presDomain,tempDomain,uVelDomain,vVelDomain):
    """
    Inflow and top BC's do not change
    Wall u, v and T do not change but pressure does
    Outflow all values float
    """
    # Wall
    presDomain[1:,0] = 2*presDomain[1:,1] - presDomain[1:,2]
    # Outflow
    presDomain[-1,1:-1] = 2*presDomain[-2,1:-1] - presDomain[-3,1:-1]
    tempDomain[-1,1:-1] = 2*tempDomain[-2,1:-1] - tempDomain[-3,1:-1]
    uVelDomain[-1,1:-1] = 2*uVelDomain[-2,1:-1] - uVelDomain[-3,1:-1]
    vVelDomain[-1,1:-1] = 2*vVelDomain[-2,1:-1] - vVelDomain[-3,1:-1]
    return presDomain,tempDomain,uVelDomain,vVelDomain

def updateAdiabaticBoundaryCond(presDomain,tempDomain,uVelDomain,vVelDomain):
    """
    Inflow and top BC's do not change
    Wall u, v and T do not change but pressure does
    Outflow all values float
    """
    # Wall
    presDomain[1:,0] = 2*presDomain[1:,1] - presDomain[1:,2]
    tempDomain[1:,0] = tempDomain[1:,1] # no heat exchange between wall and flow
    # Outflow
    presDomain[-1,1:-1] = 2*presDomain[-2,1:-1] - presDomain[-3,1:-1]
    tempDomain[-1,1:-1] = 2*tempDomain[-2,1:-1] - tempDomain[-3,1:-1]
    uVelDomain[-1,1:-1] = 2*uVelDomain[-2,1:-1] - uVelDomain[-3,1:-1]
    vVelDomain[-1,1:-1] = 2*vVelDomain[-2,1:-1] - vVelDomain[-3,1:-1]
    return presDomain,tempDomain,uVelDomain,vVelDomain
########################
#Time Step Calculations#
########################
def calcTimeStep(presDomain,tempDomain,uVelDomain,vVelDomain,
    muDomain,deltax,deltay,gamma,Pr,R,K):
    rhoDomain = presDomain/(R*tempDomain)
    vPrime =( (4/3) * muDomain * (gamma * muDomain/Pr) ) / rhoDomain
    vPrime = vPrime.max()
    a = (gamma*R*tempDomain)**0.5
    deltatCFL =( (abs(uVelDomain)/deltax) + (abs(vVelDomain)/deltay) +
        a * ( (1/deltax**2) + (1/deltay**2) )**0.5
        + 2* vPrime*((1/deltax**2)+(1/deltay**2)) ) ** -1
    deltat = K*deltatCFL.min()
    return deltat
###############################
#Predictor and Corrector Steps#
###############################
def predictor(U1,U2,U3,U5,E1,E2,E3,E5,F1,F2,F3,F5,deltat,deltax,deltay):
    xMax,yMax = U1.shape
    U1Bar = np.zeros(U1.shape)
    U2Bar = np.zeros(U1.shape)
    U3Bar = np.zeros(U1.shape)
    U5Bar = np.zeros(U1.shape)
    # Loop through all innner nodes
    for i in range(0,xMax):
        for j in range(0,yMax):
            # Boundary nodes are just carried over
            if i==0 or i==xMax-1 or j==0 or j==yMax-1:
                U1Bar[i,j] = U1[i,j]*1
                U2Bar[i,j] = U2[i,j]*1
                U3Bar[i,j] = U3[i,j]*1
                U5Bar[i,j] = U5[i,j]*1
            # Inner nodes are calculated
            else:
                U1Bar[i,j] = U1[i,j] - (deltat/deltax)*(E1[i+1,j]-E1[i,j]) - (deltat/deltay)*(F1[i,j+1]-F1[i,j])
                U2Bar[i,j] = U2[i,j] - (deltat/deltax)*(E2[i+1,j]-E2[i,j]) - (deltat/deltay)*(F2[i,j+1]-F2[i,j])
                U3Bar[i,j] = U3[i,j] - (deltat/deltax)*(E3[i+1,j]-E3[i,j]) - (deltat/deltay)*(F3[i,j+1]-F3[i,j])
                U5Bar[i,j] = U5[i,j] - (deltat/deltax)*(E5[i+1,j]-E5[i,j]) - (deltat/deltay)*(F5[i,j+1]-F5[i,j])
    return U1Bar,U2Bar,U3Bar,U5Bar

def corrector(U1,U2,U3,U5,U1Bar,U2Bar,U3Bar,U5Bar,E1Bar,E2Bar,
    E3Bar,E5Bar,F1Bar,F2Bar,F3Bar,F5Bar,deltat,deltax,deltay):
    xMax,yMax = U1.shape
    U1New = np.zeros(U1.shape)
    U2New = np.zeros(U1.shape)
    U3New = np.zeros(U1.shape)
    U5New = np.zeros(U1.shape)
    # Loop through all innner nodes
    for i in range(0,xMax):
        for j in range(0,yMax):
            # Boundary nodes are just carried over
            if i==0 or i==xMax-1 or j==0 or j==yMax-1:
                U1New[i,j] = U1Bar[i,j]*1
                U2New[i,j] = U2Bar[i,j]*1
                U3New[i,j] = U3Bar[i,j]*1
                U5New[i,j] = U5Bar[i,j]*1
            # Inner nodes are calculated
            else:
                U1New[i,j] = 0.5*( U1[i,j] + U1Bar[i,j] -
                    (deltat/deltax)*(E1Bar[i,j]-E1Bar[i-1,j]) -
                     (deltat/deltay)*(F1Bar[i,j]-F1Bar[i,j-1])  )
                U2New[i,j] = 0.5*( U2[i,j] + U2Bar[i,j] -
                    (deltat/deltax)*(E2Bar[i,j]-E2Bar[i-1,j]) -
                    (deltat/deltay)*(F2Bar[i,j]-F2Bar[i,j-1])  )
                U3New[i,j] = 0.5*( U3[i,j] + U3Bar[i,j] -
                    (deltat/deltax)*(E3Bar[i,j]-E3Bar[i-1,j]) -
                    (deltat/deltay)*(F3Bar[i,j]-F3Bar[i,j-1])  )
                U5New[i,j] = 0.5*( U5[i,j] + U5Bar[i,j] -
                    (deltat/deltax)*(E5Bar[i,j]-E5Bar[i-1,j]) -
                    (deltat/deltay)*(F5Bar[i,j]-F5Bar[i,j-1])  )
    return U1New,U2New,U3New,U5New
##################
#Iteration Scheme#
##################
def solveFlow(presDomain,tempDomain,uVelDomain,vVelDomain,
    muDomain,deltat,deltay,deltax,cp,cv,R,tempRef,muRef,adiabatic):
    """
    We will be using the MacCormack method to solve the
    Navier-Stokes equation.
    """
    ################
    #Predictor Step#
    ################
    # Calculate tau and q for predictor both F and E
    tauxyE = getTauxyfromPrim(uVelDomain,vVelDomain,muDomain,1,deltay,deltax) # predictor for E values (case1:getTauxyfromPrim)
    tauxyF = getTauxyfromPrim(uVelDomain,vVelDomain,muDomain,2,deltay,deltax) # predictor for F values (case2:getTauxyfromPrim)
    tauxxE = getTauxx(uVelDomain,vVelDomain,muDomain,1,deltay,deltax) # predictor for E values (case1:getTauxx)
    tauyyF = getTauyy(uVelDomain,vVelDomain,muDomain,1,deltay,deltax) # predictor for F values (case1:getTauyy)
    qxE = getqxE(tempDomain,cp,muDomain,deltax,1) # predictor for E values (case1:getqxE)
    qyF = getqyF(tempDomain,cp,muDomain,deltay,1) # predictor for F values (case1:getqyF)
    # Calculate the values for U, E and F arrays
    U1,U2,U3,U5 = getUfromPrim(presDomain,tempDomain,uVelDomain,
        vVelDomain,muDomain,R,cv)
    E1,E2,E3,E5 = getE(tauxyE,tauxyF,tauxxE,tauyyF,presDomain,
        tempDomain,uVelDomain,vVelDomain,qxE,R,cv)
    F1,F2,F3,F5 = getF(tauxyE,tauxyF,tauxxE,tauyyF,presDomain,
        tempDomain,uVelDomain,vVelDomain,qyF,R,cv)
    # Calculate predictor step
    U1Bar,U2Bar,U3Bar,U5Bar = predictor(U1,U2,U3,U5,E1,E2,E3,E5,
        F1,F2,F3,F5,deltat,deltax,deltay)
    # Extract primitives
    presDomain,tempDomain,uVelDomain,vVelDomain = U2Prim(
        U1Bar,U2Bar,U3Bar,U5Bar,R,cv)
    # Update BC's
    if adiabatic == False:
        presDomain,tempDomain,uVelDomain,vVelDomain = updateBoundaryCond(
            presDomain,tempDomain,uVelDomain,vVelDomain)
    else:
        presDomain,tempDomain,uVelDomain,vVelDomain = updateAdiabaticBoundaryCond(
            presDomain,tempDomain,uVelDomain,vVelDomain)
    ################
    #Corrector Step#
    ################
    muDomain = calcMu(muRef,tempDomain,tempRef) # Recalculate mu base on predicted temp
    tauxyE = getTauxyfromPrim(uVelDomain,vVelDomain,muDomain,3,deltay,deltax) # corrector for E values (case3:getTauxyfromPrim)
    tauxyF = getTauxyfromPrim(uVelDomain,vVelDomain,muDomain,4,deltay,deltax) # corrector for F values (case4:getTauxyfromPrim)
    tauxxE = getTauxx(uVelDomain,vVelDomain,muDomain,2,deltay,deltax) # corrector for E values (case2:getTauxx)
    tauyyF = getTauyy(uVelDomain,vVelDomain,muDomain,2,deltay,deltax) # corrector for F values (case2:getTauyy)
    qxE = getqxE(tempDomain,cp,muDomain,deltax,2)
    qyF = getqyF(tempDomain,cp,muDomain,deltay,2)
    U1Bar,U2Bar,U3Bar,U5Bar = getUfromPrim(presDomain,tempDomain,uVelDomain,
        vVelDomain,muDomain,R,cv)
    E1Bar,E2Bar,E3Bar,E5Bar = getE(tauxyE,tauxyF,tauxxE,tauyyF,presDomain,
        tempDomain,uVelDomain,vVelDomain,qxE,R,cv)
    F1Bar,F2Bar,F3Bar,F5Bar = getF(tauxyE,tauxyF,tauxxE,tauyyF,presDomain,
        tempDomain,uVelDomain,vVelDomain,qyF,R,cv)
    # Rearward step
    U1New,U2New,U3New,U5New =  corrector(U1,U2,U3,U5,U1Bar,U2Bar,U3Bar,U5Bar,
        E1Bar,E2Bar,E3Bar,E5Bar,F1Bar,F2Bar,F3Bar,F5Bar,deltat,deltax,deltay)
    presDomain,tempDomain,uVelDomain,vVelDomain = U2Prim(U1New,U2New,U3New,U5New,R,cv)
    # Update BC's
    if adiabatic == False:
        presDomain,tempDomain,uVelDomain,vVelDomain = updateBoundaryCond(
            presDomain,tempDomain,uVelDomain,vVelDomain)
    else:
        presDomain,tempDomain,uVelDomain,vVelDomain = updateAdiabaticBoundaryCond(
            presDomain,tempDomain,uVelDomain,vVelDomain)
    return presDomain,tempDomain,uVelDomain,vVelDomain
#############
#MAIN SCRIPT#
#############
def superVisc(lengthOfPlate,girdPtsX,girdPtsY,machInf,TwTInf,residualTarget,K,adiabatic):
    """
    This program uses an iterative method to solve
    the full Navier-Stokes equation for super sonic
    flow over a flat plate.
    Input Parameters:
    girdPtsX    - Nodes in x
    girdPtsY    - Nodes in y
    machInf     - Inlet mach number
    TwTInf      - Relationship of wall temp over free stream
    iters       - iterations to complete
    K           - Courant number
    """
    # Set flow/geometry parameters
    aInf         = 340.28        # Speed of sound: m/s
    pressureInf  = 101325.0      # Pressure N/m^2
    tempInf      = 288.16        # Kelvin
    gamma        = 1.4           # Ratio of specific heats
    Pr           = 0.71          # Prandtl number
    muRef        = 1.7894*10**-5  # Dynamic viscosity reference kg/(m*s)
    tempRef      = 288.16        # Reference temperature Kelvin
    R            = 287           # Gas constant Number J/(kg*K)
    cv           = R/(gamma-1)
    cp           = gamma * cv
    rhoInf       = pressureInf/(R*tempInf)
    uVelInf      = machInf*aInf
    vVelInf      = 0
    muInf        = calcMu(muRef,tempInf,tempRef)
    ReL          = rhoInf*uVelInf*lengthOfPlate/muInf
    eInf         = cv*tempInf
    kInf         = muInf*cp/Pr
    tempWall     = TwTInf*tempInf
    # Calculate delta x
    deltax = lengthOfPlate/(girdPtsX-1)

    # Calculate delta y
    delta = (5*lengthOfPlate)/(ReL**.5)
    heightOfGrid = 5*delta
    print(heightOfGrid)
    deltay = heightOfGrid/(girdPtsY-1)
    print(deltay)
    print('GRID DIMENSIONS','\nDelta in x: ',deltax,'\nDelta in y: ',deltay)

    # Initialize free stream
    presDomain = np.ones([girdPtsX,girdPtsY])*pressureInf
    tempDomain = np.ones([girdPtsX,girdPtsY])*tempInf
    uVelDomain = np.ones([girdPtsX,girdPtsY])*uVelInf
    vVelDomain = np.ones([girdPtsX,girdPtsY])*vVelInf
    # Initialize BC's
    presDomain,tempDomain,uVelDomain,vVelDomain = initBoundaryCond(
        presDomain,tempDomain,uVelDomain,vVelDomain,tempWall)

    # Iteration Parameters
    iteration = 1
    residual = np.array([.1])
    # Begin Simulations
    while residual[iteration-1] > residualTarget or iteration<100:
        rhoOld = presDomain/(R*tempDomain)
        # Calculate time step
        muDomain = calcMu(muRef,tempDomain,tempRef)
        deltat = calcTimeStep(presDomain,tempDomain,uVelDomain,
            vVelDomain,muDomain,deltax,deltay,gamma,Pr,R,K)
        #Solve internal points
        presDomain,tempDomain,uVelDomain,vVelDomain = solveFlow(
            presDomain,tempDomain,uVelDomain,vVelDomain,muDomain,
            deltat,deltay,deltax,cp,cv,R,tempRef,muRef,adiabatic)
        rhoNew = presDomain/(R*tempDomain)
        residuals = rhoOld - rhoNew
        residual = np.append(residual,residuals.max())
        print(
            'Iter: %4.i | Residual: %.3E | Max Pressure: %.2f | Max Temperature: %.2f | Max X-Vel: %.2f | Max Y_Vel: %.2f'
             % (iteration,residual[iteration],presDomain.max(),tempDomain.max(),uVelDomain.max(),vVelDomain.max()))
        iteration = iteration + 1
    return presDomain,tempDomain,uVelDomain,vVelDomain,residual,deltax,deltay
####################
#Plotting Functions#
####################
def plotFlow(pres,temp,uVel,vVel, lengthOfPlate, girdPtsX, heightOfPlate, girdPtsY):
    """
    Plotting script to show flowfiled
    """
    # Pressure
    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.contourf(pres.transpose())
    plt.colorbar()
    plt.title('Pressure')

    # Temperature
    plt.subplot(2,2,2)
    plt.contourf(temp.transpose())
    plt.colorbar()
    plt.title('Temp')
    # U Vel
    plt.subplot(2,2,3)
    plt.contourf(uVel.transpose())
    plt.colorbar()
    plt.title('U Vel')
    # V Vel
    plt.subplot(2,2,4)
    plt.contourf(vVel.transpose())
    plt.colorbar()
    plt.title('V Vel')

    fig.tight_layout()
    plt.savefig("flowfile.svg")
    return
def plotParam(param):
    plt.figure()
    plt.contourf(param.transpose())
    plt.colorbar()
    plt.savefig("param.png")
    return

def main():
    # User Input
    adiabatic = False
    girdPtsX=90
    girdPtsY=90
    machInf=4
    TwTInf=1
    residualTarget=10**-6
    corantNumber = 0.9
    lengthOfPlate =0.00001
    # Run main CFD code
    presDomain,tempDomain,uVelDomain,vVelDomain,residual,deltax,deltay = superVisc(
        lengthOfPlate,girdPtsX,girdPtsY,machInf,TwTInf,residualTarget,corantNumber,adiabatic)
    # Residual Plot
    plt.plot(residual)
    plt.grid(True)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Residual')
    plt.savefig("Iteration.png")
    # Preview Flow
    plotFlow(presDomain,tempDomain,uVelDomain,vVelDomain, lengthOfPlate, girdPtsX, 8.189266555235586e-06, girdPtsY)
    # Save data
    save = input('Save? (y/n)')
    if save == 'y':
        fileName = input('File Name:') + '.h5'
        h5f = h5py.File(fileName, 'w')
        h5f.create_dataset('pres', data=presDomain)
        h5f.create_dataset('temp', data=tempDomain)
        h5f.create_dataset('u', data=uVelDomain)
        h5f.create_dataset('v', data=vVelDomain)
        h5f.create_dataset('residual', data=residual)
        h5f.create_dataset('lengthOfPlate',data = lengthOfPlate)
        h5f.create_dataset('deltax',data=deltax)
        h5f.create_dataset('deltay',data=deltay)
        h5f.close()
        print(fileName, 'saved')
    pass

main()
