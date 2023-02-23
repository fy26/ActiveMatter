#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:26:28 2022

@author: fanyang
"""
import numpy as np

from matplotlib import pyplot, cm, colors
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse, special
import scipy.sparse.linalg as sp
import scipy.special as spe 


import math
import time

def laplace(A, dx, dy, dxp, dyp):
    #centered scheme for a Laplace equation
    
    la = (((A[1:-1, 2:] - A[1:-1, 1:-1]) / dx[1:-1, 2:] - (A[1:-1, 1:-1] - A[1:-1, 0:-2]) / dx[1:-1, 1:-1])
          / dxp +
          ((A[2:, 1:-1] - A[1:-1, 1:-1]) / dy[2:, 1:-1] - (A[1:-1, 1:-1] - A[0:-2, 1:-1]) / dy[1:-1, 1:-1])
          / dyp)
    
    return la

def pxpy(A, X, Y):
    #\partial^2 A / \partial x \partialy
    pd = ((A[2:, 2:] - A[2:, 0:-2]) / (X[2:, 2:] - X[2:, 0:-2]) 
          - (A[0:-2, 2:] - A[0:-2, 0:-2]) / (X[0:-2, 2:] - X[0:-2, 0:-2])) / (Y[2:, 1:-1] - Y[0:-2, 1:-1])
    
    return pd
 
def BC_noflux(A, X, Y):
    #second-order accuracy no-flux boundary conditions
    A[:, 0] = ((A[:, 1] * (X[:, 2] - X[:, 0])**2 - A[:, 2] * (X[:, 1] - X[:, 0])**2) 
               / ((X[:, 2] - X[:, 0])**2 - (X[:, 1] - X[:, 0])**2))
    
    A[:, -1] = ((A[:, -2] * (X[:, -3] - X[:, -1])**2 - A[:, -3] * (X[:, -2] - X[:, -1])**2)
                / ((X[:, -3] - X[:, -1])**2 - (X[:, -2] - X[:, -1])**2))
    
    A[0, :] = ((A[1, :] * (Y[2, :] - Y[0, :])**2 - A[2, :] * (Y[1, :] - Y[0, :])**2) 
               / ((Y[2, :] - Y[0, :])**2 - (Y[1, :] - Y[0, :])**2))
    
    A[-1, :] = ((A[-2, :] * (Y[-3, :] - Y[-1, :])**2 - A[-3, :] * (Y[-2, :] - Y[-1, :])**2) 
               / ((Y[-3, :] - Y[-1, :])**2 - (Y[-2, :] - Y[-1, :])**2))
    
    return A    

def build_up_A(c, dx, dy, d2x, d2y, delta, gamma, ux, uy, ster, sigmapxx, sigmapxy, gf, cf):
    
    A =( 1 / d2x * (delta[1:-1, 2:] - delta[1:-1, 0:-2])
         - 1 / d2x * (ster[1:-1, 2:] - ster[1:-1, 0:-2])       
          + gamma * c[1:-1, 1:-1] * ux[1:-1, 1:-1] 
          + gf[1:-1, 1:-1] * cf[1:-1, 1:-1] * ux[1:-1, 1:-1]
        + 1 / d2x * (sigmapxx[1: -1, 2:] - sigmapxx[1: -1, 0:-2]) 
         + 1 / d2y * (sigmapxy[2:, 1:-1] - sigmapxy[0:-2, 1:-1]))
         
    return A

def build_up_vcoeff(gamma, c, dx, dy, dxp, dyp, etac, gf, cf):

    B = 1 / 2 / dxp / dx[1:-1, 2:]  * (etac[1:-1, 1:-1] + etac[1:-1, 2:])
    C = 1 / 2 / dxp / dx[1:-1, 1:-1]  * (etac[1:-1, 1:-1] + etac[1:-1, 0:-2])
    D = 1 / 2 / dyp / dy[2:, 1:-1] * (etac[1:-1, 1:-1] + etac[2:, 1:-1]) 
    E = 1 / 2 / dyp / dy[1:-1, 1:-1] * (etac[1:-1, 1:-1] + etac[0:-2, 1:-1])
    F = gamma * c[1:-1, 1:-1]  + 2 * B + 2 * C + D + E + gf[1:-1, 1:-1] * cf[1:-1, 1:-1]
    L = gamma * c[1:-1, 1:-1]  + B + C + 2 * D + 2 * E + gf[1:-1, 1:-1] * cf[1:-1, 1:-1]
    return [B, C, D, E, F, L]

def build_up_J(d2x, d2y, etac):

    
    Jm10 = etac[1:-1, 0:-2] / d2x / d2y
    Jp10 = etac[1:-1, 2:] / d2x / d2y
    
    J0m1 = etac[0:-2, 1:-1] / d2x / d2y
    J0p1 = etac[2:, 1:-1] / d2x / d2y
    
    return [Jm10, Jp10, J0m1, J0p1]

def build_up_G(c, dx, dy, d2x, d2y, delta, gamma, ux, uy, ster,
                sigmapyy, sigmapxy, gf, cf):
    
    G = ( 1 / d2y * (delta[2:, 1:-1] - delta[0:-2, 1:-1])
         - 1 / d2y * (ster[2:, 1:-1] - ster[0:-2, 1:-1]) 
         + gamma * c[1:-1, 1:-1] * uy[1:-1, 1:-1]           
          +  gf[1:-1, 1:-1] * cf[1:-1, 1:-1] * uy[1:-1, 1:-1]
         + 1 / d2x * (sigmapxy[1: -1, 2:] - sigmapxy[1: -1, 0:-2]) 
         + 1 / d2y * (sigmapyy[2:, 1:-1] - sigmapyy[0:-2, 1:-1]))
    return G



def build_up_Z(c, zeta, gf, gamma, cf, psi, bf):
    # I is the inverse of Z / zeta
    Zxx = ((1 + psi * gamma * cf / (bf * c + gamma)) * zeta 
           / (1 + zeta * (c + gf / gamma * cf)))
    Zyy = Zxx.copy()
    Zxy = np.zeros_like(Zxx)
    
    return [Zxx, Zyy, Zxy]


def build_up_ZV(c, vx, vy, gf, gamma, cf, psi, bf, zeta):
    
    Zc1 = ((1 + psi * gamma * cf / (bf * c + gamma)) * zeta * (c + gf / gamma * cf)
           / (1 + zeta * (c + gf / gamma * cf)))
    Zc2 = c * psi * (1 + bf * cf / (bf * c + gamma))
    Zc = Zc1 + Zc2
    ZVx = Zc * vx
    ZVy = Zc * vy  
    
    return [ZVx, ZVy]
def build_up_pcoeff(Zxx, Zxy, Zyy, dx, dy, dxp, dyp, d2x, d2y):
    H = 1 / 2 / dxp / dx[1:-1, 2:] * (Zxx[1:-1, 1:-1] + Zxx[1:-1, 2:])
    M = 1 / 2 / dxp / dx[1:-1, 1:-1] * (Zxx[1:-1, 1:-1] + Zxx[1:-1, 0:-2])
    N = 1 / 2 / dyp / dy[2:, 1:-1] * (Zyy[2:, 1:-1] + Zyy[1:-1, 1:-1])
    O = 1 / 2 / dyp / dy[1:-1, 1:-1] * (Zyy[0:-2, 1:-1] + Zyy[1:-1, 1:-1])
    R = 1 / d2y / d2x * (Zxy[0:-2, 1:-1] + Zxy[1:-1, 0:-2])
    W = 1 / d2y / d2x * (Zxy[2:, 1:-1] + Zxy[1:-1, 0:-2])
    SS = 1 / d2y / d2x * (Zxy[0:-2, 1:-1] + Zxy[1:-1, 2:])
    TT = 1 / d2y / d2x * (Zxy[2:, 1:-1] + Zxy[1:-1, 2:])

    return [H, M, N, O, R, W, SS, TT]

def solve_linear_V(Wx, Sx, Px, Nx, Ex, LLx, ULx, PPx, LRx, URx, Qx,
                    Wy, Sy, Py, Ny, Ey, LLy, ULy, PPy, LRy, URy, Qy):
    
    nj, ni = Qx.shape 
    n = ni * nj
    Vx = np.zeros((nj+2, ni+2))
    Vy = np.zeros((nj+2, ni+2))
    
    Sxc = Sx.copy()
    Nxc = Nx.copy()
    LLxc = LLx.copy()
    ULxc = ULx.copy()
    LRxc = LRx.copy()
    URxc = URx.copy()

    
    Syc = Sy.copy()
    Nyc = Ny.copy()
    LLyc = LLy.copy()
    ULyc = ULy.copy()
    LRyc = LRy.copy()
    URyc = URy.copy()
    
    
    ## build matrix 1
    ## no-slip BCs
    Sxc[0, : ] = 0
    Nxc[-1, :] = 0
  
    
    ## create diagonals
    ma1_mnj = Wx[:, 1:].flatten('F') # diagonal at -n_j
    ma1_m1 = np.delete(Sxc.flatten('F'), 0) # diagonal at -1
    ma1_0 = Px.flatten('F') # diagonal at 0
    ma1_1 = np.delete(Nxc.flatten('F'), -1) # diagonal at 1
    ma1_nj = Ex[:, 0:-1].flatten('F') # diagonal at n_j
    
    ## build matrix 4
    ## no-slip BCs
    Syc[0, : ] = 0
    Nyc[-1, :] = 0
    
    ## create diagonals
    ma4_mnj = Wy[:, 1:].flatten('F') # diagonal at -n_j
    ma4_m1 = np.delete(Syc.flatten('F'), 0) # diagonal at -1
    ma4_0 = Py.flatten('F') # diagonal at 0
    ma4_1 = np.delete(Nyc.flatten('F'), -1) # diagonal at 1
    ma4_nj = Ey[:, 0:-1].flatten('F') # diagonal at n_j
    
    # build matrix 2
    # no-slip BCs
    LLxc[0, : ] = 0
    ULxc[-1, :] = 0
    LRxc[0, : ] = 0
    URxc[-1, :] = 0
    
    ma2_mnjm1 = np.delete(LLxc[:, 1:].flatten('F'), 0) # diagonal at -n_j-1
    ma2_mnj1 = np.append(0, ULxc[:, 1:].flatten('F')) # diagonal at -n_j+1
    ma2_0 = PPx.flatten('F') # diagonal at 0
    ma2_njm1 = np.append(LRxc[:, 0:-1].flatten('F'), 0) # diagonal at n_j-1
    ma2_nj1= np.delete(URxc[:, 0:-1].flatten('F'), -1) # diagonal at n_j+1
    
    # build matrix 3
    # no-slip BCs
    LLyc[0, : ] = 0
    ULyc[-1, :] = 0
    LRyc[0, : ] = 0
    URyc[-1, :] = 0
    
    ma3_mnjm1 = np.delete(LLyc[:, 1:].flatten('F'), 0) # diagonal at -n_j-1
    ma3_mnj1 = np.append(0, ULyc[:, 1:].flatten('F')) # diagonal at -n_j+1
    ma3_0 = PPy.flatten('F') # diagonal at 0
    ma3_njm1 = np.append(LRyc[:, 0:-1].flatten('F'), 0) # diagonal at n_j-1
    ma3_nj1= np.delete(URyc[:, 0:-1].flatten('F'), -1) # diagonal at n_j+1
    #create the overall matrix
    
    #matrix 1 and 4
    d_0 = np.concatenate((ma1_0, ma4_0))
    d_m1 = np.concatenate((ma1_m1,[0], ma4_m1))
    d_1 = np.concatenate((ma1_1, [0], ma4_1))
    d_mnj = np.concatenate((ma1_mnj, np.zeros(nj), ma4_mnj))
    d_nj = np.concatenate((ma1_nj, np.zeros(nj), ma4_nj))
    
    #matrix 2    
    d_n = ma2_0
    d_npnjm1 = ma2_njm1
    d_npnjp1 = ma2_nj1
    d_nmnjp1 = np.concatenate((np.zeros(nj-1), ma2_mnj1, np.zeros(nj-1)))
    d_nmnjm1 = np.concatenate((np.zeros(nj+1), ma2_mnjm1, np.zeros(nj+1)))
    
    #matrix 3    
    d_mn = ma3_0
    d_mnmnjp1 = ma3_mnj1
    d_mnmnjm1 = ma3_mnjm1
    d_mnpnjp1 = np.concatenate((np.zeros(nj+1), ma3_nj1, np.zeros(nj+1)))
    d_mnpnjm1 = np.concatenate((np.zeros(nj-1), ma3_njm1, np.zeros(nj-1)))
    

    diagonals = [d_mnmnjm1, d_mnmnjp1, d_mn, d_mnpnjm1, d_mnpnjp1, d_mnj, d_m1, d_0, d_1, d_nj, 
                 d_nmnjm1, d_nmnjp1, d_n, d_npnjm1, d_npnjp1]

    A = sparse.diags(diagonals, [-n-nj-1, -n-nj+1, -n, -n+nj-1, -n+nj+1, -nj, -1, 0, 1, nj,
                                n-nj-1, n-nj+1, n, n+nj-1, n+nj+1])
    Q = np.concatenate((Qx.flatten('F'), Qy.flatten('F')))
    ## solve using LU decomposition of sparse matrix
    sln = sp.splu(A).solve(Q)
    Vx[1:-1, 1:-1] = sln[0:n].reshape((nj, ni), order = 'F') 
    Vy[1:-1, 1:-1] = sln[n:].reshape((nj, ni), order = 'F') 
    return [Vx, Vy]


def solve_V(c, dx, dy, dxp, dyp, d2x, d2y, gamma, eta,
            ux, uy, X, Y, dc, xis, sigmapxx, sigmapxy, sigmapyy, gf, cf, ca):
    
    etac = eta * dc * c 
    delta = dc * c * ca
    ster = xis * c * (c + cf)

    A = build_up_A(c, dx, dy, d2x, d2y, delta, gamma, ux, uy, 
               ster, sigmapxx, sigmapxy, gf, cf)
    
    B, C, D, E, F, L = build_up_vcoeff(gamma, c, dx, dy, dxp, dyp, etac, gf, cf)

    Jm10, Jp10, J0m1, J0p1 = build_up_J(d2x, d2y, etac)

    
    G = build_up_G(c, dx, dy, d2x, d2y, delta, gamma, ux, uy, ster,
                sigmapyy, sigmapxy, gf, cf)
    
    place_holder = np.zeros_like(Jm10)

    vxn, vyn = solve_linear_V(2*C, E, -F, D, 2*B, J0m1, -J0p1, place_holder, 
                                   -J0m1, J0p1, -A,
                                   C, 2*E, -L, 2*D, B, J0m1, -J0m1, place_holder,
                                   -Jp10, Jp10, -G)
    return [vxn, vyn]

def solve_linear_pressure(W, S, P, N, E, LL, UL, LR, UR, Q, X, Y):
                    
    
    nj, ni = Q.shape 
    
    pr = np.zeros((nj+2, ni+2))
    
    Sc = S.copy()
    Nc = N.copy()
    Pc = P.copy()
    Wc = W.copy()
    Ec = E.copy()
    LLc = LL.copy()
    ULc = UL.copy()
    LRc = LR.copy()
    URc = UR.copy()
    
    ## zero-gradient BCs
    
    Sc[:, 0] += LL[:, 0]
    Nc[:, 0] += UL[:, 0]
    
    
    Sc[:, -1] += LR[:, -1]
    Nc[:, -1] += UR[:, -1]
    
    
    Wc[0, :] += LL[0, :]
    Ec[0, :] += LR[0, :]
    
    
    Wc[-1, :] += UL[-1, :]
    Ec[-1, :] += UR[-1, :]
    
    Pc[:, 0] += Wc[:, 0]
    Pc[:, -1] += Ec[:, -1]
    Pc[0, :] += Sc[0, :]
    Pc[-1, :] += Nc[-1, :]
    
    Sc[0, :] = 0
    LLc[0, :] = 0
    LRc[0, :] = 0
    
    Nc[-1, :] = 0 
    ULc[-1, :] = 0
    URc[-1, :] = 0
    
    ## create diagonals
    d_mnj = Wc[:, 1:].flatten('F') # diagonal at -n_j
    d_m1 = np.delete(Sc.flatten('F'), 0) # diagonal at -1
    d_0 = Pc.flatten('F') # diagonal at 0
    d_1 = np.delete(Nc.flatten('F'), -1) # diagonal at 1
    d_nj = Ec[:, 0:-1].flatten('F') # diagonal at n_j
    d_mnjm1 = np.delete(LLc[:, 1:].flatten('F'), 0) # diagonal at -n_j-1
    d_mnj1 = np.append(0, ULc[:, 1:].flatten('F')) # diagonal at -n_j+1
    d_njm1 = np.append(LRc[:, 0:-1].flatten('F'), 0) # diagonal at n_j-1
    d_nj1= np.delete(URc[:, 0:-1].flatten('F'), -1) # diagonal at n_j+1
    
    diagonals = [d_mnjm1, d_mnj, d_mnj1, d_m1, d_0, d_1, d_njm1, d_nj, 
                 d_nj1]
    
    A = sparse.diags(diagonals, [-nj-1, -nj, -nj+1, -1, 0, 1, nj-1,
                                nj, nj+1])
    
    ## solve using LU decomposition of sparse matrix
    pr[1:-1, 1:-1] = sp.splu(A).solve(Q.flatten('F')).reshape((nj, ni), order = 'F')
    
    return BC_noflux(pr, X, Y)
    
    
    
def iter_V(c, dx, dy, dxp, dyp, d2x, d2y, gamma, vx, vy, ux, uy, X, Y, dc, xis, 
            eta, zeta, sigmapxx, sigmapxy, sigmapyy, gf, cf, ca, psi, bf):
           
    if not c.any():
        
        return [vx, vy, ux, uy]
    
    else:
        
        ## calculate microtubule velocity V        
        
        vxn, vyn = solve_V(c, dx, dy, dxp, dyp, d2x, d2y, gamma, eta,
            ux, uy, X, Y, dc, xis, sigmapxx, sigmapxy, sigmapyy, gf, cf, ca)
       
       
    
        ## calculate pressure 
        Zxx, Zyy, Zxy = build_up_Z(c, zeta, gf, gamma, cf, psi, bf)
    
        
        ZVx, ZVy = build_up_ZV(c, vx, vy, gf, gamma, cf, psi, bf, zeta)
        
        H, M, N, O, R, W, SS, TT = build_up_pcoeff(Zxx, Zxy, Zyy, dx, dy, dxp, dyp, d2x, d2y)
        
        Qp = (ZVx[1:-1, 2:] - ZVx[1:-1, 0:-2]) / d2x + (ZVy[2:, 1:-1] - ZVy[0:-2, 1:-1]) / d2y

        p = solve_linear_pressure(M, O, -H-M-N-O, N, H, R, -W, -SS, TT, Qp, X, Y)

        ## calculate fluid velocity U
        uxn = np.zeros_like(ux)
        uyn = np.zeros_like(uy)
        cv = c[1:-1, 1:-1] + gf[1:-1, 1:-1] / gamma * cf[1:-1, 1:-1]
        dpdx = (p[1:-1, 2:] - p[1:-1, 0:-2])/ d2x 
        dpdy = (p[2:, 1:-1] - p[0:-2, 1:-1])/ d2y
        
        uxn[1:-1, 1:-1] = zeta * (cv * vx[1:-1, 1:-1] - dpdx) / (1 + zeta * cv) 
        uyn[1:-1, 1:-1] = zeta * (cv * vy[1:-1, 1:-1] - dpdy) / (1 + zeta * cv) 
        
        return [vxn, vyn, uxn, uyn]
        

def build_up_A_2(c, dx, dy, d2x, d2y, delta, gamma, ux, uy, 
               ster, sigmapxx, sigmapxy, gf, cf, J0m1, J0p1, vy):
    
    A =( 1 / d2x * (delta[1:-1, 2:] - delta[1:-1, 0:-2])
         - 1 / d2x * (ster[1:-1, 2:] - ster[1:-1, 0:-2])       
          + gamma * c[1:-1, 1:-1] * ux[1:-1, 1:-1] 
          + gf[1:-1, 1:-1] * cf[1:-1, 1:-1] * ux[1:-1, 1:-1]
        + 1 / d2x * (sigmapxx[1: -1, 2:] - sigmapxx[1: -1, 0:-2]) 
         + 1 / d2y * (sigmapxy[2:, 1:-1] - sigmapxy[0:-2, 1:-1])
       + J0m1 * vy[0:-2, 0:-2] - J0p1 * vy[2:, 0:-2] - J0m1 * vy[0:-2, 2:] + J0p1 * vy[2:, 2:])
         
    return A

def build_up_G_2(c, dx, dy, d2x, d2y, delta, gamma, ux, uy, ster,
                sigmapyy, sigmapxy, gf, cf,  Jm10, Jp10, vx):
    
    G = ( 1 / d2y * (delta[2:, 1:-1] - delta[0:-2, 1:-1])
         - 1 / d2y * (ster[2:, 1:-1] - ster[0:-2, 1:-1]) 
         + gamma * c[1:-1, 1:-1] * uy[1:-1, 1:-1]           
          +  gf[1:-1, 1:-1] * cf[1:-1, 1:-1] * uy[1:-1, 1:-1]
         + 1 / d2x * (sigmapxy[1: -1, 2:] - sigmapxy[1: -1, 0:-2]) 
         + 1 / d2y * (sigmapyy[2:, 1:-1] - sigmapyy[0:-2, 1:-1])
        + Jm10 * vx[0:-2, 0:-2] - Jm10 * vx[2:, 0:-2] - Jp10 * vx[0:-2, 2:] + Jp10 * vx[2:, 2:])
    return G

def solve_linear_V_2(Wx, Sx, Px, Nx, Ex, Qx):
    
    nj, ni = Qx.shape 
    n = ni * nj
    V = np.zeros((nj+2, ni+2))
    
    Sxc = Sx.copy()
    Nxc = Nx.copy()

    
    ## no-slip BCs
    Sxc[0, : ] = 0
    Nxc[-1, :] = 0
  
    
    ## create diagonals
    d_mnj = Wx[:, 1:].flatten('F') # diagonal at -n_j
    d_m1 = np.delete(Sxc.flatten('F'), 0) # diagonal at -1
    d_0 = Px.flatten('F') # diagonal at 0
    d_1 = np.delete(Nxc.flatten('F'), -1) # diagonal at 1
    d_nj = Ex[:, 0:-1].flatten('F') # diagonal at n_j
    
    diagonals = [d_mnj, d_m1, d_0, d_1, d_nj]

    A = sparse.diags(diagonals, [-nj, -1, 0, 1, nj])
    Q = Qx.flatten('F')
    ## solve using LU decomposition of sparse matrix
    sln = sp.splu(A).solve(Q)
    V[1:-1, 1:-1] = sln[0:n].reshape((nj, ni), order = 'F') 
    
    return V
    

def solve_V_2(c, dx, dy, dxp, dyp, d2x, d2y, gamma, eta,
            ux, uy, X, Y, dc, xis, sigmapxx, sigmapxy, sigmapyy, gf, cf, ca, vy):
    
    etac = eta * dc * c 
    delta = dc * c * ca
    ster = xis * c * c
    
    Jm10, Jp10, J0m1, J0p1 = build_up_J(d2x, d2y, etac)

    A = build_up_A_2(c, dx, dy, d2x, d2y, delta, gamma, ux, uy, 
               ster, sigmapxx, sigmapxy, gf, cf, J0m1, J0p1, vy)
    
    
    B, C, D, E, F, L = build_up_vcoeff(gamma, c, dx, dy, dxp, dyp, etac, gf, cf)

    vxn = solve_linear_V_2(2*C, E, -F, D, 2*B, -A)

    G = build_up_G_2(c, dx, dy, d2x, d2y, delta, gamma, ux, uy, ster,
                sigmapyy, sigmapxy, gf, cf, Jm10, Jp10, vxn)
    

    vyn = solve_linear_V_2(C, 2*E, -L, 2*D, B, -G)
    return [vxn, vyn]

    
def iter_V_2(c, dx, dy, dxp, dyp, d2x, d2y, gamma, vx, vy, ux, uy, X, Y, dc, xis, 
            eta, zeta, sigmapxx, sigmapxy, sigmapyy, gf, cf, ca, psi, bf):
           
    if not c.any():
        
        return [vx, vy, ux, uy]
    
    else:
        
        ## calculate microtubule velocity V        
        
        vxn, vyn = solve_V_2(c, dx, dy, dxp, dyp, d2x, d2y, gamma, eta,
            ux, uy, X, Y, dc, xis, sigmapxx, sigmapxy, sigmapyy, gf, cf, ca, vy)
       
       
    
        ## calculate pressure 
        Zxx, Zyy, Zxy = build_up_Z(c, zeta, gf, gamma, cf, psi, bf)
    
        
        ZVx, ZVy = build_up_ZV(c, vx, vy, gf, gamma, cf, psi, bf, zeta)
        
        H, M, N, O, R, W, SS, TT = build_up_pcoeff(Zxx, Zxy, Zyy, dx, dy, dxp, dyp, d2x, d2y)
        
        Qp = (ZVx[1:-1, 2:] - ZVx[1:-1, 0:-2]) / d2x + (ZVy[2:, 1:-1] - ZVy[0:-2, 1:-1]) / d2y

        p = solve_linear_pressure(M, O, -H-M-N-O, N, H, R, -W, -SS, TT, Qp, X, Y)

        ## calculate fluid velocity U
        uxn = np.zeros_like(ux)
        uyn = np.zeros_like(uy)
        cv = c[1:-1, 1:-1] + gf[1:-1, 1:-1] / gamma * cf[1:-1, 1:-1]
        dpdx = (p[1:-1, 2:] - p[1:-1, 0:-2])/ d2x 
        dpdy = (p[2:, 1:-1] - p[0:-2, 1:-1])/ d2y
        
        uxn[1:-1, 1:-1] = zeta * (cv * vx[1:-1, 1:-1] - dpdx) / (1 + zeta * cv) 
        uyn[1:-1, 1:-1] = zeta * (cv * vy[1:-1, 1:-1] - dpdy) / (1 + zeta * cv) 
        
        return [vxn, vyn, uxn, uyn]

        

    
    
    
    
    
def vorticity(ux, uy, d2x, d2y):
    vo = -(ux[2:, 1:-1] - ux[0:-2, 1:-1]) / d2y + (uy[1:-1, 2:] - uy[1:-1, 0:-2]) / d2x
    return vo

def update_vf(c, vx, vy, gamma, bf, ux, uy, d2x, d2y):
    vfx = (bf * c * vx + gamma * ux) / (bf * c + gamma)
    vfy = (bf * c * vy + gamma * uy) / (bf * c + gamma)
    return [vfx, vfy]

def timestep_m(df, m, ux, uy, pd, pm, pemi, dx, dy, dxp, dyp, d2x, d2y, dt, X, Y):
    mn = np.zeros_like(m)
    mn[1:-1, 1:-1] = (m[1:-1, 1:-1] + 2 * pm * dt * df[1:-1, 1:-1] - 2 * dt * pd[1:-1, 1:-1] * m[1:-1, 1:-1]**2 - 
                     (m[2:, 1:-1] * uy[2:, 1:-1] - 
                        m[0:-2, 1:-1] * uy[0:-2, 1:-1]) / d2y * dt - 
                       (m[1:-1, 2:]  * ux[1:-1, 2:] - 
                        m[1:-1, 0:-2] * ux[1:-1, 0:-2]) / d2x * dt + 
                      pemi * dt * laplace(m, dx, dy, dxp, dyp))
                     
    ## applying BCs
   
    mn = BC_noflux(mn, X, Y)
    
    return mn

def timestep_df(df, c, cf, ux, uy, m, poff, pn1, pn2, pedi, dx, dy, dxp, dyp, d2x, 
                d2y, dt, pm, pd, X, Y, dc):
    dfn = np.zeros_like(df)
    dfn[1:-1, 1:-1] = (df[1:-1, 1:-1] 
                       + poff * dt * dc[1:-1, 1:-1]
                       - dt * pn1 * cf[1:-1, 1:-1] * df[1:-1, 1:-1] -
                       dt * pn2 * c[1:-1, 1:-1] * df[1:-1, 1:-1] 
                        -
                     (df[2:, 1:-1] * uy[2:, 1:-1] - 
                        df[0:-2, 1:-1] * uy[0:-2, 1:-1]) / d2y * dt - 
                       (df[1:-1, 2:]  * ux[1:-1, 2:] - 
                        df[1:-1, 0:-2] * ux[1:-1, 0:-2]) / d2x * dt + 
                      pedi * dt * laplace(df, dx, dy, dxp, dyp)
                     - pm * dt * df[1:-1, 1:-1] + dt * pd[1:-1, 1:-1] * m[1:-1, 1:-1]**2)
                     
    ## applying BCs
    dfn = BC_noflux(dfn, X, Y) #second-order accuracy
    
    return dfn

def timestep_dc(df, c, cf, vx, vy, poff, pn1, pn2, dx, dy, dxp, 
                dyp, d2x, d2y, dt, X, Y, dc):
    dcn = np.zeros_like(dc)
    dcn[1:-1, 1:-1] = (dc[1:-1, 1:-1]
                       - poff * dt * dc[1:-1, 1:-1]
                       + dt * pn1 * cf[1:-1, 1:-1] * df[1:-1, 1:-1] +
                       dt * pn2 * c[1:-1, 1:-1] * df[1:-1, 1:-1] 
                     - (dc[2:, 1:-1] * vy[2:, 1:-1] - 
                        dc[0:-2, 1:-1] * vy[0:-2, 1:-1]) / d2y * dt - 
                       (dc[1:-1, 2:]  * vx[1:-1, 2:] - 
                        dc[1:-1, 0:-2] * vx[1:-1, 0:-2]) / d2x * dt)
                     
    ## applying BCs
    dcn = BC_noflux(dcn, X, Y) #second-order accuracy
    
    return dcn

def timestep_cf(c, cf, df, vfx, vfy, pu, pn1, dx, dy, dxp, dyp, d2x, d2y, dt, X, Y, pei):
    cfn = np.zeros_like(cf)
    cfn[1:-1, 1:-1] = (cf[1:-1, 1:-1]
                      - (cf[2:, 1:-1] * vfy[2:, 1:-1] - cf[0:-2, 1:-1] * vfy[0:-2, 1:-1] ) / d2y * dt - 
                       (cf[1:-1, 2:] * vfx[1:-1, 2:] - cf[1:-1, 0:-2] * vfx[1:-1, 0:-2]) / d2x * dt
                       - dt * pn1 * cf[1:-1, 1:-1] * df[1:-1, 1:-1]
                       + dt * pu * c[1:-1, 1:-1] + pei * dt * laplace(cf, dx, dy, dxp, dyp))
    ## applying BCs
    
    cfn = BC_noflux(cfn, X, Y)
    return cfn

    
def timestep_c(c, cf, df, vx, vy, pu, pn1, d2x, d2y, dt, X, Y):
    cn = np.zeros_like(c)
    cn[1:-1, 1:-1] = (c[1:-1, 1:-1]
                      - (c[2:, 1:-1] * vy[2:, 1:-1] - c[0:-2, 1:-1] * vy[0:-2, 1:-1] ) / d2y * dt - 
                       (c[1:-1, 2:] * vx[1:-1, 2:] - c[1:-1, 0:-2] * vx[1:-1, 0:-2]) / d2x * dt
                       + dt * pn1 * cf[1:-1, 1:-1] * df[1:-1, 1:-1]
                       - dt * pu * c[1:-1, 1:-1])
    ## applying BCs
    
    cn = BC_noflux(cn, X, Y)
    return cn

def timestep_atp(ca, dc, ux, uy, ka, dx, dy, dxp, dyp, d2x, d2y, dt, X, Y, peai):
    can = np.zeros_like(ca)
    can[1:-1, 1:-1] = (ca[1:-1, 1:-1]
                      - (ca[2:, 1:-1] * uy[2:, 1:-1] - ca[0:-2, 1:-1] * uy[0:-2, 1:-1] ) / d2y * dt - 
                       (ca[1:-1, 2:] * ux[1:-1, 2:] - ca[1:-1, 0:-2] * ux[1:-1, 0:-2]) / d2x * dt
                       - dt * ka * ca[1:-1, 1:-1] * dc[1:-1, 1:-1] 
                      + peai * dt * laplace(ca, dx, dy, dxp, dyp))
    ## applying BCs
    
    can = BC_noflux(can, X, Y)
    return can



 

def timestep_sigmap(sigmapxx, sigmapxy, sigmapyy, vx, vy, dx, dy, dc, d2x, d2y, dt, X, Y, sp, c, pu):
    sigmapxxn = np.zeros_like(sigmapxx)
    sigmapxxn[1:-1, 1:-1] = ((1 - pu * dt) * sigmapxx[1:-1, 1:-1] -
                             vy[1:-1, 1:-1] * (sigmapxx[2:, 1:-1] - sigmapxx[0:-2, 1:-1]) / d2y * dt - 
                              vx[1:-1, 1:-1] * (sigmapxx[1:-1, 2:] - sigmapxx[1:-1, 0:-2]) / d2x * dt
                      + dt * (2 * sigmapxx[1:-1, 1:-1] * (vx[1:-1, 2:] - vx[1:-1, 0:-2]) / d2x
                                 + 2 * sigmapxy[1:-1, 1:-1] * ((vx[2:, 1:-1] - vx[0:-2, 1:-1]) / d2y))
                      + 2 * sp * dc[1:-1, 1:-1] * c[1:-1, 1:-1] * dt * (vx[1:-1, 2:] - vx[1:-1, 0:-2]) / d2x)
                      
    
    sigmapyyn = np.zeros_like(sigmapyy)
    sigmapyyn[1:-1, 1:-1] = ((1 - pu * dt) * sigmapyy[1:-1, 1:-1] -
                             vy[1:-1, 1:-1] * (sigmapyy[2:, 1:-1] - sigmapyy[0:-2, 1:-1]) / d2y * dt - 
                              vx[1:-1, 1:-1] * (sigmapyy[1:-1, 2:] - sigmapyy[1:-1, 0:-2]) / d2x * dt
                      + dt * (2 * sigmapyy[1:-1, 1:-1] * (vy[2:, 1:-1] - vy[0:-2, 1:-1]) / d2y
                                 + 2 * sigmapxy[1:-1, 1:-1] * ((vy[1:-1, 2:] - vy[1:-1, 0:-2]) / d2x))
                      + 2 * sp * dc[1:-1, 1:-1] * c[1:-1, 1:-1] *  dt * (vy[2:, 1:-1] - vy[0:-2, 1:-1]) / d2y )
    
    sigmapxyn = np.zeros_like(sigmapxy)
    sigmapxyn[1:-1, 1:-1] = ((1 - pu * dt) * sigmapxy[1:-1, 1:-1] -
                             vy[1:-1, 1:-1] * (sigmapxy[2:, 1:-1] - sigmapxy[0:-2, 1:-1]) / d2y * dt - 
                              vx[1:-1, 1:-1] * (sigmapxy[1:-1, 2:] - sigmapxy[1:-1, 0:-2]) / d2x * dt
                      + dt * (sigmapyy[1:-1, 1:-1] * (vx[2:, 1:-1] - vx[0:-2, 1:-1]) / d2y
                                 + sigmapxy[1:-1, 1:-1] * ((vy[2:, 1:-1] - vy[0:-2, 1:-1]) / d2y +  (vx[1:-1, 2:] - vx[1:-1, 0:-2]) / d2x)
                                  + sigmapxx[1:-1, 1:-1] * (vy[1:-1, 2:] - vy[1:-1, 0:-2]) / d2x)
                      + sp * dt * dc[1:-1, 1:-1] * c[1:-1, 1:-1] *  ((vy[1:-1, 2:] - vy[1:-1, 0:-2]) / d2x +  (vx[2:, 1:-1] - vx[0:-2, 1:-1]) / d2y))
                     
                     
    ## applying BCs
    sigmapxxn = BC_noflux(sigmapxxn, X, Y) #second-order accuracy
    sigmapyyn = BC_noflux(sigmapyyn, X, Y)
    sigmapxyn = BC_noflux(sigmapxyn, X, Y)
    return [sigmapxxn, sigmapyyn, sigmapxyn]
                                      
def light_rec(AS, X, Y, cx, cy, rx, ry, bx, by, theta, theta0):
    t_rot = theta - theta0
    XX = X * math.cos(t_rot) - Y * math.sin(t_rot)
    YY = X * math.sin(t_rot) + Y * math.cos(t_rot)
    S_rectangle = AS / (1 + ((XX-cx) / rx)** (2 * bx)) / (1 + ((YY-cy) / ry)** (2 * by))
    return S_rectangle

def light_circ(AS, X, Y, cx, cy, r, b):
    S_circ = AS / (1 + (((X - cx)**2 + (Y - cy)**2) / r**2)**b)
    return S_circ