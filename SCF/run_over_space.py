# -*- coding: utf-8 -*-
#coding=utf-8
import numpy as np
import sys
import struct
from io import StringIO
import os
import math
import random
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import copy

IsDouble=False
batch_size=28

G=-1.0
bond_set=5
Size=4
Eps={}
e1=1.0
e2=1.0
volume_ratio=6.0
Eps['AA']=e1
Eps['AB']=e1
Eps['BB']=e1
Eps['AAb']=e1/volume_ratio
Eps['BAb']=e1
Eps['AbAb']=e1/volume_ratio/volume_ratio
eps_N_A=10.0
N_Ab = 100.0
bond_number=2.0
K_bond=None
ds=0.02
omega_para=0.0
d_np=6.32/12*2
d=1.12/12*2
bond_length = 2.0
L = bond_length*Size
dr = 0.1
dr3=dr*dr*dr
N = int((L + 0.01) / dr)
rx = np.linspace(0, L - dr, N)
ry = np.linspace(0, L - dr, N)
rz = np.linspace(0, L - dr, N)
r = np.array(np.meshgrid(rx, ry, rz))
r = r.transpose(3, 2, 1, 0).reshape(N, N, N, 3)
d_m = d_np / 2.0 + d / 2.0
sigma=d_m/(2.0**(1.0/6.0))
chain_l = 4.0



def period_bond_connect(pos, b_length, box_size):
    dis=np.linalg.norm((pos[:, None, :]-pos[None, :, :]+0.5*box_size) % box_size-0.5*box_size, axis=-1)
    con=(np.abs(np.triu(dis)-b_length)<1e-5)
    i,j=np.where(con)
    return np.stack((i,j),axis=1)



def Cal_Distance(r1,r2,L):
    return np.linalg.norm((r1[:, None, :] - r2[None, :, :] + 0.5 * L) % L - 0.5 * L, axis=-1)
    pass

def Get_NP_Potential(r,r_Np,L):
    K=K_bond
    bond_R=d_m
    dis=Cal_Distance(r.reshape(N*N*N,3),r_Np[None,:],L)+1e-12
    dis=dis.reshape(N,N,N)
    potential_WCA=np.zeros((N,N,N))
    # potential_bond=np.zeros((N,N,N))
    contact=dis<d_m
    # d_cut2=d_m/3.0*5
    # contact2=dis<d_cut2
    u = 0.5 * K * ((dis - bond_R) ** 2)
    # bond_area=(~contact)&contact2
    p_shift=4*((sigma/d_m)**12-(sigma/d_m)**6)
    # p_shift2=4*eps_N_A*((sigma/d_cut2)**12-(sigma/d_cut2)**6)
    potential_WCA[contact]=4*((sigma/dis[contact])**12-(sigma/dis[contact])**6)-p_shift
    potential_WCA[potential_WCA>200]=200.0
    u[u>200]=200.0
    potential_Ab=u+potential_WCA
    # potential_Ab[contact2]=4*eps_N_A*((sigma/dis[contact2])**12-(sigma/dis[contact2])**6)-p_shift2
    # bond_area=potential_Ab<0.0
    return potential_WCA,potential_Ab,u,contact



def Init_field_gen(shift):
    # fix np position
    chain_num=192
    Np_pos_id=np.array([N//2,N//2,N//2])
    dr_MAX=r[Np_pos_id[0]+11,Np_pos_id[1]+11,Np_pos_id[2]+11,:]-r[Np_pos_id[0],Np_pos_id[1],Np_pos_id[2],:]
    Np_pos=r[Np_pos_id[0],Np_pos_id[1],Np_pos_id[2],:]+shift*dr_MAX

    V_WCA,V_bond,u,contact=Get_NP_Potential(r,Np_pos,L)

    rho_old_A = np.zeros((N, N, N)).astype(float) + 1.0 * chain_num * chain_l / N / N / N / dr / dr / dr
    rho_old_Ab = np.zeros((N, N, N)).astype(float) + N_Ab / N / N / N / dr / dr / dr
    rho_old_A[contact] = 0.0
    rho_old_Ab[contact] = 0.0
    rho_old_A = rho_old_A / np.sum(rho_old_A) * 1.0 * chain_num * chain_l / dr / dr / dr
    rho_old_Ab = rho_old_Ab / np.sum(rho_old_Ab) * N_Ab / dr / dr / dr
    omega_A = Eps['AA'] * rho_old_A + Eps['AAb'] * rho_old_Ab + V_WCA
    omega_Ab = Eps['AAb'] * rho_old_A + Eps['AbAb'] * (rho_old_Ab) + V_WCA



    if IsDouble:
        omega_A.reshape(-1, 1).tofile('./omega_Net_init_D.dat')
        omega_Ab.reshape(-1, 1).tofile('./omega_Ab_init_D.dat')
        V_WCA.reshape(-1, 1).tofile('./WCA_D.dat')
        u.reshape(-1, 1).tofile('.}/u_D.dat')
    else:
        omega_A.astype(np.float32).reshape(-1, 1).tofile('./omega_Net_init.dat')
        omega_Ab.astype(np.float32).reshape(-1, 1).tofile('./omega_Ab_init.dat')
        V_WCA.astype(np.float32).reshape(-1, 1).tofile('./WCA.dat')
        u.astype(np.float32).reshape(-1, 1).tofile('./u.dat')
    return 0



c_pos=np.array([1.0,1.0,1.0])
count=0
r_l = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.333333],
    [0.0, 0.0, 0.666667],
    [0.0, 0.0, 1.0],
    [0.0, 0.333333, 0.333333],
    [0.0, 0.333333, 0.666667],
    [0.0, 0.333333, 1.0],
    [0.0, 0.666667, 0.666667],
    [0.0, 0.666667, 1.0],
    [0.0, 1.0, 1.0],
    [0.333333, 0.333333, 0.333333],
    [0.333333, 0.333333, 0.666667],
    [0.333333, 0.333333, 1.0],
    [0.333333, 0.666667, 0.666667],
    [0.333333, 0.666667, 1.0],
    [0.333333, 1.0, 1.0],
    [0.666667, 0.666667, 0.666667],
    [0.666667, 0.666667, 1.0],
    [0.666667, 1.0, 1.0],
    [1.0, 1.0, 1.0],
]
K_bond_l=[100.0]
print("start")
run_i=0
for r_ in r_l:
    rx = r_[0]
    ry = r_[1]
    rz = r_[2]
    shift = np.array([rx, ry, rz])
    dis = shift * 1.0 - c_pos
    dis = np.sum(dis * dis) ** 0.5
    if dis > d_m:
        for K_ in K_bond_l:
            K_bond = K_
            Init_field_gen(shift=shift)
            os.system(
                "python3 run_one_pos.py -x {} -y {} -z {} -K {} -v {} -i {}".format(rx, ry, rz, K_bond,
                                                                                    volume_ratio, run_i))


