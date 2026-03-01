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
from optparse import OptionParser
optParser = OptionParser()

optParser.add_option("-x","--xaxis", action="store", type=float, dest="x",default=0.0, help="xaxis")
optParser.add_option("-y","--yaxis", action="store", type=float, dest="y",default=0.0, help="yaxis")
optParser.add_option("-z","--zaxis", action="store", type=float, dest="z",default=0.0, help="zaxis")
optParser.add_option("-K","--K_bond", action="store", type=float, dest="K",default=100.0, help="K_bond")
optParser.add_option("-v","--volume_ratio", action="store", type=float, dest="volume_ratio",default=100.0, help="volume_ratio")
optParser.add_option("-i","--run_i", action="store", type=int, dest="run_i",default=0, help="run_i")
options,args=optParser.parse_args()
sx=options.x
sy=options.y
sz=options.z
K=options.K
run_i=options.run_i
volume_ratio=options.volume_ratio
antibody_phi=[10**-4.0] #
antibody_phi=[10**-4.5,10**-4.25,10**-4.0,10**-3.75,10**-3.5,10**-3.25,10**-3.0,10**-2.5,10**-2.0,10**-1.5,10**-1.25,10**-1.0]
len_Ab=len(antibody_phi)
# antibody_phi=[10**-4.25,10**-4.0,10**-3.75,10**-3.5,10**-3.25,10**-3.0,10**-2.5,10**-2.0,10**-1.5,10**-1.25,10**-1.0]
lx=47.99803543
r = 2. ** (1. / 6) / 2.
antibody_num=np.zeros(len(antibody_phi))
for i in range(len(antibody_phi)):
    antibody_num[i] = int(antibody_phi[i] * lx * lx * lx / (4. / 3. * np.pi * r * r * r) / 2)
Istest=False
# Ab_l=[5.0,10.0,20.0,50.0,100.0]   #1.0,2.0,5.0,10.0,20.0,50.0,100.0,200.0,500.0,1000.0,
Ab_l=antibody_num.astype(float).tolist()
# k_NP=[-4.75]
k_NP=[-2.75,-3.0,-3.5,-4.0,-4.5,-5.0,-5.5,-5.75]

len_k=len(k_NP)
dG_NP_l=[]
dG_NP_ex=6.549
dG_cnf_ex_l=[]
if abs(K-200.0)<0.1:
    dG_NP_ex+=-0.419
    dG_cnf_ex_l += [0.1886,1.0008]
elif abs(K-50.0)<0.1:
    dG_NP_ex+=0.249
    dG_cnf_ex_l += [0.4035,1.3685]
else:
    dG_NP_ex += 0.1
    dG_cnf_ex_l += [0.0]
# dG_cnf_ex_l=[0.0,0.2,0.5]
for k_ in k_NP:
    dg_np=dG_NP_ex+np.log((10**k_)/0.05)-np.log(10.)
    dG_NP_l+=[dg_np]
#2-2000,
# dG_NP_l=[-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0]  #-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,


#,-4--10
set_bond_l=[-1.0]
set_bond=-1.0
dG_net_l=[-7.684] #-7.6
# dG_net_l=[200.0] #-7.6
dG_net=-1.0
step=0.2
iteration=30
bij_iteration=1000
M=20
# K=100.0
# os.system('kill -STOP 9678')
run_list=[[-4.25,-5.75],[-4.0,-5.75],[-3.5,-5.75],[-3.0,-5.75],[-2.5,-5.75],[-2.0,-5.75],[-1.5,-5.75],[-1.25,-5.75],[-1.0,-5.75]]
run_list=[[-4.5,-5.75],[-4.5,-5.5],[-4.5,-5.0],[-4.5,-4.5],[-4.5,-4.0],[-4.5,-3.5],[-4.5,-3.0],[-4.5,-2.75]]
date='V5_Fit250917'
i_net = 0
run_now=-1
for dG_net in dG_net_l:
    i_np = 0
    for dG_NP in dG_NP_l:
        i_ab=0
        for Ab in Ab_l:
            for dG_cnf_ex in dG_cnf_ex_l:
                a=antibody_phi[i_ab]
                b=k_NP[i_np]
                a=np.log10(a)
                is_run=False
                for rl in run_list:
                    if abs(rl[0]-a)<0.01 and abs(rl[1]-b)<0.01:
                        is_run=True
                        break
                if not is_run:
                    continue
                name = '{}x{}y{}z{}Ab{}dG{}cnfEx{}Net{}M{}K{}V{}'.format(date,sx,sy,sz,antibody_phi[i_ab],k_NP[i_np],dG_cnf_ex,dG_net,M,K,volume_ratio)
                if not os.path.exists('./{}'.format(name)):
                    os.system("mkdir {}".format(name))
                    os.system(
                        "./kernel_M20_V5 {} {} {} {} {} {} {} {} {}".format(Ab, set_bond, dG_NP, dG_net, step, iteration,
                                                                            bij_iteration, dG_cnf_ex, volume_ratio))
                    os.system("cp ./F.dat ./{}".format(name))
                    os.system("cp ./P_m.dat ./{}".format(name))
                    os.system("cp ./P_ubind_m.dat ./{}".format(name))
                    os.system("cp ./z_m.dat ./{}".format(name))
                    os.system("cp ./Ab0.dat ./{}".format(name))
                    os.system("cp ./Ab1.dat ./{}".format(name))
                    os.system("cp ./Net_state.dat ./{}".format(name))
                    os.system("cp ./p_i_m.dat ./{}".format(name))
                    i_b=0
                    while i_b<min(Ab,M)+0.1:
                        os.system("cp ./bij_{}.dat ./{}".format(i_b,name))
                        os.system("cp ./bi_{}.dat ./{}".format(i_b,name))
                        i_b+=1
                    os.system("cp ./rhoNet__{}.dat ./{}".format(iteration-1,name))
                    os.system("cp ./rhoAb__{}.dat ./{}".format(iteration-1,name))
                    os.system("cp ./P_cross__{}.dat ./{}".format(iteration-1,name))
                    os.system("cp ./omegaNet__{}.dat ./{}".format(iteration-1,name))
                    os.system("cp ./p_chain149_cid42_it{}.dat ./{}".format(iteration-1,name))
                    os.system("cp ./p_chain148_cid42_it{}.dat ./{}".format(iteration-1,name))
                    os.system("cp ./p_chain147_cid42_it{}.dat ./{}".format(iteration-1,name))
                    os.system("cp ./p_chain144_cid42_it{}.dat ./{}".format(iteration-1,name))
                    os.system("cp ./p_chain136_cid42_it{}.dat ./{}".format(iteration-1,name))
                    os.system("cp ./p_chain101_cid42_it{}.dat ./{}".format(iteration-1,name))
                    os.system("cp ./p_chain144_cid41_it{}.dat ./{}".format(iteration-1,name))
            i_ab+=1
        i_np+=1
    i_net+=1
