# -*- coding: utf-8 -*-
#coding=utf-8
from scipy.special import comb, perm, factorial
import numpy as np
import sys
from io import StringIO
import os
import math
import random
import matplotlib.pyplot as plt
# from mayavi import mlab
import scipy.io as sio
import time
import copy
from scipy import special
from mpmath import *
import argparse

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--D', type=float, default=0.007158304101921)
parser.add_argument('--k1', type=float, default=0.000493579855788648)
parser.add_argument('--k2', type=float, default=0.004005304747939)
parser.add_argument('--gama', type=float, default=0.0025)
parser.add_argument('--end_r', type=float, default=10.0)
parser.add_argument('--t0', type=float, default=100.0)
parser.add_argument('--date', type=str, default='250917T1')

args = parser.parse_args()

# 赋值给同名变量
D = args.D
k1 = args.k1
k2 = args.k2
gama = args.gama
end_r = args.end_r
t0 = args.t0
date = args.date

mp.dps = 50
mp.pretty = True
dG=-2.3
D1=D
f_k=2.0
F_N=50
f_s=0.0
f_r=0.0001
f_t=0.1
def fun_kt(k):
    global f_k
    f_k=k
    result=invertlaplace(fun_L2, f_t, method='talbot')
    print(k)
    return result*sin(f_r*f_k)*f_k
def fun_F_rt(tt):
    global f_t
    f_t=tt
    result=quad(fun_kt,[0, inf])/np.pi/np.pi/2/f_r
    return result
def fun_L2(s):
    B = D * f_k * f_k / 2 / gama
    A = 2 * gama
    m1 = k1 / (k1 + k2)
    m2 = k2 / (k1 + k2)
    F = 0.0
    n = 0
    # Fi=[]
    a = (s + k1) / A
    b = a + 1
    try:
        F = 1 / a * hyper([a], [b], B) / A
    except:
        F = 1 / a * hyper([a], [b], B, maxterms=10 ^ 600000) / A
    F = F * exp(-B)
    result = m2 * F + (m1 + k1 * k2 * m2 * F * F + k1 * m2 * F + k2 * m1 * F) / (s + D * f_k * f_k + k2 - k1 * k2 * F)
    return result
def fun_Gau(s):
    result=1/(D*f_k*f_k+s)
    return result



Num_r=100
rF_l=np.arange(0.0+1e-5,end_r+1e-5,end_r/Num_r)
t=np.array([t0])
# rfll=[]
# for tt in t:
#     rfll+=[rF_l*(tt**0.5)/(t[0]**0.5)]
f_rt=np.zeros((rF_l.shape[0],t.shape[0]))
tii=0
i=0
for tt in t:
    rii = 0
    # rF_l=rfll[tii]
    for rr in rF_l:
        f_r=rr
        ft=fun_F_rt(tt)
        print(ft)
        f_rt[rii,tii]=ft
        rii+=1
        sio.savemat('data{}t{}.mat'.format(date,t0), {'f_rt': f_rt, 't': t, 'r': rF_l}, do_compression=True)
        if ft<1e-9:
            break
        if rii % 10 == 0:
            print(rii)
    tii+=1

sio.savemat('data{}t{}.mat'.format(date, t0), {'f_rt': f_rt, 't': t, 'r': rF_l}, do_compression=True)

debug=1





