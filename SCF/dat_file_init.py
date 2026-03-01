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
# from mayavi import mlab
import scipy.io as sio
import time
import copy

IsDouble=False
All_cross_free=True
batch_size=28

G=-1.0
bond_set=5
Size=4
Eps={}
e1=1.0
e2=-1.0
Eps['AA']=e1
Eps['AB']=e1
Eps['BB']=e1
Eps['AAb']=e1
Eps['BAb']=e2
Eps['AbAb']=e1
eps_N_A=0.0
bond_number=2.0
ds=0.02
omega_para=0.0
al_modify=True
# d_np=2.0
d_np=1.0
d=0.2
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
Fs=1/dr
kx = np.linspace(-N/2, N/2-1, N)/N*np.pi*2*Fs
ky = np.linspace(-N/2, N/2-1, N)/N*np.pi*2*Fs
kz = np.linspace(-N/2, N/2-1, N)/N*np.pi*2*Fs
k_fft = np.array(np.meshgrid(kx, ky, kz))
k_fft = k_fft.transpose(3, 2, 1, 0).reshape(N, N, N, 3)
dss=ds/2
kout=np.exp(-ds / 6.0 * (k_fft[:, :, :, 0] ** 2 + k_fft[:, :, :, 1] ** 2 + k_fft[:, :, :, 2] ** 2))
kout2=np.exp(-dss / 6.0 * (k_fft[:, :, :, 0] ** 2 + k_fft[:, :, :, 1] ** 2 + k_fft[:, :, :, 2] ** 2))
kout=np.fft.ifftshift(kout)
kout2=np.fft.ifftshift(kout2)
kout=kout.reshape(-1,1)
kout2=kout2.reshape(-1,1)

def period_bond_connect(pos, b_length, box_size):
    dis=np.linalg.norm((pos[:, None, :]-pos[None, :, :]+0.5*box_size) % box_size-0.5*box_size, axis=-1)
    con=(np.abs(np.triu(dis)-b_length)<1e-5)
    i,j=np.where(con)
    return np.stack((i,j),axis=1)


def Eta_s(s):
    if (s >= chain_l/2-0.5) & (s < chain_l/2+0.5):
        return 1.0
    else:
        return 0.0


def Cal_Distance(r1,r2,L):
    return np.linalg.norm((r1[:, None, :] - r2[None, :, :] + 0.5 * L) % L - 0.5 * L, axis=-1)
    pass

def Get_NP_Potential(r,r_Np,L):
    K=100.0
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
    V_WCA=V_WCA*0.0
    u=u*0.0
    rho_old_A = np.zeros((N, N, N)).astype(float) + 1.0 * chain_num * chain_l / N / N / N / dr / dr / dr
    omega_A = Eps['AA'] * rho_old_A

    if IsDouble:
        omega_A.reshape(-1, 1).tofile('./omega_Net_init_D.dat')
        V_WCA.reshape(-1, 1).tofile('./WCA_D.dat')
        u.reshape(-1, 1).tofile('./u_D.dat')
    else:
        omega_A.astype(np.float32).reshape(-1, 1).tofile('./omega_Net_init.dat')
        V_WCA.astype(np.float32).reshape(-1, 1).tofile('./WCA.dat')
        u.astype(np.float32).reshape(-1, 1).tofile('./u.dat')
    return 0

kk = []
for i in range(Size):
    kk += [(2 * i + 1) / (Size * 2)]
r_crosslink = []
for i in kk:
    for j in kk:
        for k in kk:
            r_crosslink += [np.array([int(i * N), int(j * N), int(k * N)])]
r_crosslink = np.array(r_crosslink)
rc_1d = r_crosslink[:, 0] * N * N + r_crosslink[:, 1] * N + r_crosslink[:, 2]
linkpoint_num = r_crosslink.shape[0]
chains = period_bond_connect(np.array([r[jj[0], jj[1], jj[2], :] for jj in r_crosslink]), bond_length, L)
center_pos=r[N//2,N//2,N//2]
pos=np.array([r[jj[0], jj[1], jj[2], :] for jj in r_crosslink])
dis=np.linalg.norm((pos[:, None, :]-center_pos[None, None, :]+0.5*L) % L-0.5*L, axis=-1).flatten()
c=chains.reshape(-1,1).astype(np.int32)
temp=np.array([0,1])
IsFree=(dis<2.0)
if All_cross_free:
    IsFree[:]=1
IsFree=IsFree.astype(np.int32)

k=50.0
j=0
for jj in r_crosslink:
    c_pos=r[jj[0], jj[1], jj[2], :]
    dis=Cal_Distance(r.reshape(N*N*N,3),c_pos[None,:],L)+1e-12
    dis = dis.reshape(N, N, N)
    h_c=0.5*k*dis*dis
    h_c=h_c.reshape(-1,1)
    i=0
    while i<chains.shape[0]:
        chain=chains[i]
        if chain[0]==j:
            if IsDouble:
                h_c.tofile('h1_{}_D.dat'.format(i))
            else:
                h_c.astype(np.float32).tofile('h1_{}.dat'.format(i))
        if chain[1]==j:
            if IsDouble:
                h_c.tofile('h2_{}_D.dat'.format(i))
            else:
                h_c.astype(np.float32).tofile('h2_{}.dat'.format(i))
        i+=1
    j+=1
if IsDouble:
    kout.tofile('k_fft_D.dat')
    kout2.tofile('k2_fft_D.dat')
else:
    kout.astype(np.float32).tofile('k_fft.dat')
    kout2.astype(np.float32).tofile('k2_fft.dat')
s = np.arange(0, chain_l, ds)
eta_s = np.zeros(s.shape[0]).astype(float)
c.tofile('chains.dat')
IsFree.tofile('cross_free.dat')
eta_s.astype(np.int32).tofile('eta_s.dat')
shift_l=[1.0]   #unit: ax
set_MA_l=[-1.0]
set_bond_l=[-1.0]  # 1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.,11.0,12.0,13.0,14.0,15.0
set_bond=-1.0
i_ma=0
set_MA=-1.0
c_pos=np.array([1.1,1.1,1.1])
count=0
for rx in [0.0]:  # ,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
    for ry in [0.0]:
        for rz in [0.0]:
            shift = np.array([rx, ry, rz])
            dis = shift * 1.1 - c_pos
            dis = np.sum(dis * dis) ** 0.5
            if dis > d_m:
                Init_field_gen(shift=shift)

