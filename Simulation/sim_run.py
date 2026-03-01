import os
import time
import pandas as pd
import numpy as np

# print('sleep 1h ')
# time.sleep(3600+60*7)
print('start')
# os.system("kill -STOP 12002")
# exit(0)
dying='dod'
phi=0.03
graft=54
# Force=np.arange(8.0, 10.1, 0.5).astype(float)

# Force=[7.0,11.0, 15.0]
# dpr=0.02
# pr=0.2
# depoly_period = 2
# depoly_Pr=np.arange(0.1, 0.6, 0.08)
# Period = [5, 1, 10, 20]
# p=5
# k_bond=[220, 300, 400, 500]
# f=8.0
rand_l=[1326]
bP=[10**-4.0,10**-3.0,10**-5.0]
rP=[0.05,0.5,0.005]
mode=['nvt']  # , 'nvt'
antibody_phi=[10**-2.0] #10**-4.0,10**-3.5,10**-3,10**-2.5,10**-2
Dp_l=[0.00005]
Pp=[0.1]*(len(Dp_l))
Dp = Dp_l[0]
m=mode[0]
netname='0304S4L00100p0.03'  # ,'0304S4L10101p0.03'
netB = netname[6:12]
n=netname
STEP_K=2.0
rn=0
date_='251212t52'
g_l=[1.0]
Is_fix=0 #
run_list=[[-3.5,-3.75],[-3.25,-3.75],[-2.25,-3.75],[-2.0,-4.0],[-3.25,-3.5],[-1.75,-4.5],[-1.5,-5.5],[-1.5,-5.25]]
for gama in g_l:
    for rand in rand_l:
        date=date_+'r{}g{}'.format(rand,gama)
        for Dp in Dp_l:
            j = 0
            while j < len(bP):
                i=0
                while i<len(antibody_phi):
                    rn+=1
                    Ab_bp=[np.log10(antibody_phi[i]),np.log10(bP[j])]
                    # if not Ab_bp in run_list:
                    #     i += 1
                    #     continue
                    sim_name = date + dying + 'g{0}p{1}Ab{2}m55S4{3}r{4}b{5}D{6}P{7}{8}'.format(graft, phi, antibody_phi[i], m, rP[j], bP[j],Dp,Pp[0],netB)
                    os.system("python3 sys_init_di.py -a {} -n {}".format(antibody_phi[i],n))
                    os.system("python3 sys_init_exist_di.py -r {0} -b {1} -m {2} -D {3} -a {4} -n {5} -P {6} -N {7} -d {8} -i {9} -S {10}".format(rP[j], bP[j], m,Dp,antibody_phi[i],n,Pp[0],date,dying,Is_fix,sim_name))
                    os.system("cp ./sim_reaction_di.py simulation/{0}".format(sim_name))
                    os.system("cd ./simulation/{0} && python2 sim_reaction_di.py -r {1} -b {2} -m {3} -D {4} -s {5} -P {6} -R {7} -g {8} -i {9}".format(sim_name, rP[j], bP[j],m,Dp,STEP_K,Pp[0],rand,gama,Is_fix))
                    os.system("cp ./analyse_function_raw_di.py simulation/{0}".format(sim_name))
                    os.system("cp ./analyse_function_distri0.py simulation/{0}".format(sim_name))
                    os.system("cp ./Galamostxmlreader.py simulation/{0}".format(sim_name))
                    os.system("cp ./delete_0xml_dcd.py simulation/{0}".format(sim_name))
                    os.system("cd ./simulation/{0} && python3 analyse_function_raw_di.py -s {1}".format(sim_name,STEP_K))
                    os.system("cd ./simulation/{0} && python3 analyse_function_distri0.py".format(sim_name))
                    os.system("cd ./simulation/{0} && python3 delete_0xml_dcd.py".format(sim_name))
                    i += 1
                j+=1

# os.system("kill -CONT 12002")