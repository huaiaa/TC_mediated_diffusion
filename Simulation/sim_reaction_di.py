#!/usr/bin/python
import sys
import numpy as np
import time
sys.path.append('/opt/galamost4/lib')  # the path where the GALAMOST program is installed
import galamost
from optparse import OptionParser

global _options
parser = OptionParser()
parser.add_option('--gpu', dest='gpu', help='GPU on which to execute')
parser.add_option("-f","--force", action="store", type=float, dest="force",
                     default=9.0, help="force_of_attract")
parser.add_option("-D","--Dp", action="store", type=float, dest="Dp",
                     default=0.0015, help="Pr of depoly")
parser.add_option("-P","--Pp", action="store", type=float, dest="Pp",
                     default=0.1, help="Pr of poly")
parser.add_option("-k","--kbond", action="store", type=int, dest="kbond",
                     default=220, help="K of bond")
parser.add_option("-R","--rand", action="store", type=int, dest="rand",
                     default=1235, help="rand")
parser.add_option("-r","--rP", action="store", type=float, dest="rP",
                     default=0.5, help="Pr of reaction AM")
parser.add_option("-b","--bP", action="store", type=float, dest="bP",
                     default=0.05, help="Pr of broken AM")
parser.add_option("-m","--mode", action="store", type=str, dest="mode",
                     default='nve', help="nvt or nve")
parser.add_option("-s", "--step", action="store", type=float, dest="step",
                  default=1.0, help="step")
parser.add_option("-g", "--gama", action="store", type=float, dest="gama",
                  default=1.0, help="gama")
parser.add_option("-i","--is_fix", action="store", type=int, dest="is_fix",
                     default=0, help="is_fix")
(_options, args) = parser.parse_args()

# print('sleep')
# time.sleep(12000)
# print('start')

filename = 'PIN_init.xml' # initial configuration file
build_method = galamost.XmlReader(filename)
perform_config = galamost.PerformConfig(_options.gpu) # GPU index
all_info = galamost.AllInfo(build_method, perform_config)
force=_options.force
Dp=_options.Dp
depoly_period=10
depoly_Pr=Dp
gama=_options.gama
Pp=_options.Pp
kbond=_options.kbond
rand=_options.rand
rP=_options.rP
bP=_options.bP
mode=_options.mode
STEP_K=_options.step
is_fix=_options.is_fix
# poly_period=int(Pp)
# poly_Pr=Pp-float(poly_period)
np.savetxt('force.txt', np.array([force]))
np.savetxt('Dp.txt', np.array([Dp]))
np.savetxt('Pp.txt', np.array([Pp]))
np.savetxt('rP.txt', np.array([rP]))
np.savetxt('bP.txt', np.array([bP]))
np.savetxt('kbond.txt', np.array([kbond]))
np.savetxt('gama.txt', np.array([gama]))

dt = 0.005
app = galamost.Application(all_info, dt)

neighbor_list = galamost.NeighborList(all_info, 3.0 ,0.1)#(,rcut, rbuffer)
neighbor_list.addExclusionsFromBodys() # remove the interactions of particles in a same body

lj = galamost.LjForce(all_info, neighbor_list,3.0)
lj.setParams('A', 'A', 1.0, 1.0, 1.0, 1.12246) #type, type, epsilon, sigma, alpha, rcut
lj.setParams('A', 'B' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('B', 'B' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('A', 'C' ,1.0 ,1.0 ,1.0, 1.12246) #type, type, epsilon, sigma, alpha, rcut
lj.setParams('B', 'C' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('C', 'C' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('A', 'D' ,1.0 ,1.0 ,1.0, 1.12246) #type, type, epsilon, sigma, alpha, rcut
lj.setParams('B', 'D' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('C', 'D' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('D', 'D' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('A', 'F' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('B', 'F' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('C', 'F' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('D', 'F' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('F', 'F' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('A', 'G' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('B', 'G' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('C', 'G' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('D', 'G' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('F', 'G' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('G', 'G' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('A', 'H' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('B', 'H' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('C', 'H' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('D', 'H' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('F', 'H' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('G', 'H' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('H', 'H' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
if is_fix==1:
    lj.setParams('A', 'R', 1.0, 1.0, 1.0, 1.12246)  # type, type, epsilon, sigma, alpha, rcut
    lj.setParams('B', 'R', 1.0, 1.0, 1.0, 1.12246)  # type, type, epsilon, sigma, alpha, rcut
    lj.setParams('C', 'R', 1.0, 1.0, 1.0, 1.12246)  # type, type, epsilon, sigma, alpha, rcut
    lj.setParams('D', 'R', 1.0, 1.0, 1.0, 1.12246)  # type, type, epsilon, sigma, alpha, rcut
    lj.setParams('F', 'R', 1.0, 1.0, 1.0, 1.12246)  # type, type, epsilon, sigma, alpha, rcut
    lj.setParams('G', 'R', 1.0, 1.0, 1.0, 1.12246)  # type, type, epsilon, sigma, alpha, rcut
    lj.setParams('H', 'R', 1.0, 1.0, 1.0, 1.12246)  # type, type, epsilon, sigma, alpha, rcut
    lj.setParams('R', 'R', 1.0, 1.0, 1.0, 1.12246)  # type, type, epsilon, sigma, alpha, rcut
lj.setEnergy_shift()#The potential is shifted to make the energy be zero at cutoff distance
app.add(lj)

all_info.addBondType('F-G')
all_info.addBondType('B-H')
bondforceharmonic=galamost.BondForceHarmonic(all_info)
bondforceharmonic.setParams('B-H', kbond, 0.96)
bondforceharmonic.setParams('F-G', kbond, 0.96)
bondforceharmonic.setParams('A-A', kbond, 0.96)
bondforceharmonic.setParams('A-B', kbond, 0.96)
bondforceharmonic.setParams('B-A', kbond, 0.96)
bondforceharmonic.setParams('G-H', kbond, 0.96)
app.add(bondforceharmonic)


# bondforcefene = galamost.BondForceFene(all_info)
# bondforcefene.setParams('A-A', 30, 1.5)
# bondforcefene.setParams('A-B', 30, 1.5)
# bondforcefene.setParams('B-A', 30, 1.5)
# bondforcefene.setParams('B-B', 30, 1.5)
# bondforcefene.setParams('F-G', 30, 1.5)
# bondforcefene.setParams('B-H', 30, 1.5)
# bondforcefene.setParams('G-H', 30, 1.5)
# bondforcefene.setParams('H-G', 30, 1.5)

# app.add(bondforcefene)
# pass
# # ##### reaction origin ##############
# # reaction = galamost.Polymerization(all_info, neighbor_list, 1.12246 ,163)
# # reaction.setFuncReactRule(True, float(kbond), 1.5, 0.960, 10.0, galamost.Polymerization.Func.harmonic)
# # # reaction.setFuncReactRule(True, 1250.000, 1.0,0.470, 10.0, galamost.Polymerization.Func.harmonic)
# # reaction.setPr(poly_Pr)
# # reaction.setMaxCris('H',1)
# # reaction.setMaxCris('B',1)
# # # sets the connected bond upper limited number.
# # reaction.setNewBondType('B-H')
# # reaction.setPeriod(poly_period)
# # app.add(reaction)
# #
# # reaction = galamost.DePolymerization(all_info, 1.0, 16361)
# # reaction.setParams('B-H', float(kbond), 1.5, 0.960, 10.0, depoly_Pr, galamost.DePolymerization.Func.harmonic)
# # # sets bondname, K, r_0, b_0, epsilon0, Pr, and function.
# # reaction.setPeriod(depoly_period)
# # # sets how many steps to react.
# # app.add(reaction)
# # Brownian dynamics for rigid bodies

# ############ new reaction #################
if Pp>0:
    reaction_Ab_M = galamost.Polymerization(all_info, 'H', 1.0, neighbor_list, 1.12246, 163)
    reaction_Ab_M.setPr('H','B',Pp)
    reaction_Ab_M.setPrFactor('H','B',1)
    reaction_Ab_M.setMaxCris('H',1)
    reaction_Ab_M.setMaxCris('B',1)
    reaction_Ab_M.setNewBondType('B-H')
    reaction_Ab_M.setFuncReactRule(False, float(kbond), 1.5, 0.960, 10.0, galamost.Polymerization.Func.harmonic)
    app.add(reaction_Ab_M)
if rP>0:
    reaction_Ab_P = galamost.Polymerization(all_info, 'G', 1.0, neighbor_list, 1.12246, 163)
    reaction_Ab_P.setPr('G','F',rP)
    reaction_Ab_P.setPrFactor('G','F',1)
    reaction_Ab_P.setMaxCris('G',1)
    reaction_Ab_P.setMaxCris('F',1)
    reaction_Ab_P.setNewBondType('F-G')
    reaction_Ab_P.setFuncReactRule(False, float(kbond), 1.5, 0.960, 10.0, galamost.Polymerization.Func.harmonic)
    app.add(reaction_Ab_P)
#
if Dp>0:
    depoly_Ab_M = galamost.DePolymerization(all_info, 1.0, 16361)
    depoly_Ab_M.setParams('B-H', float(kbond), 1.5, 0.960, 10.0, Dp, galamost.DePolymerization.Func.NoFunc)
    # sets bondname, K, r_0, b_0, epsilon0, Pr, and function.
    depoly_Ab_M.setPeriod(10)
    # sets how many steps to react.
    app.add(depoly_Ab_M)
#
if bP>0:
    depoly_Ab_P = galamost.DePolymerization(all_info, 1.0, 16361)
    depoly_Ab_P.setParams('F-G', float(kbond), 1.5, 0.960, 10.0, bP, galamost.DePolymerization.Func.NoFunc)
    # sets bondname, K, r_0, b_0, epsilon0, Pr, and function.
    depoly_Ab_P.setPeriod(depoly_period)
    # sets how many steps to react.
    app.add(depoly_Ab_P)

if mode == 'nve':
    bgroup = galamost.ParticleSet(all_info, 'body')  # the group of the body particles
    rigidnve = galamost.NveRigid(all_info, bgroup)  # nve for particle
    app.add(rigidnve)
    np.savetxt('mode.txt', np.array([0]))
else:
    bgroup = galamost.ParticleSet(all_info, 'body')  # the group of the body particles
    bdnvt_rigid = galamost.BdNvtRigid(all_info, bgroup, 1.0, 12563)  # temperature, seed for random number generator
    bdnvt_rigid.setGamma(gama)
    app.add(bdnvt_rigid)
    np.savetxt('mode.txt', np.array([1]))

list1=['A','B','G','H']
nbgroup = galamost.ParticleSet(all_info, list1)
bdnvt = galamost.BdNvt(all_info, nbgroup, 1.0, rand)
# bdnvt.setGamma(gama)
app.add(bdnvt)


group = galamost.ParticleSet(all_info,'all') # all particles
comp_info = galamost.ComputeInfo(all_info, group) # calculating system informations, such as temperature, pressure, and momentum

DInfo = galamost.DumpInfo(all_info, comp_info, 'data.log') # output system informations, such as temperature, pressure, and momentum
DInfo.setPeriod(100)
app.add(DInfo)

zm = galamost.ZeroMomentum(all_info) # remove the momentum of the center of mass
zm.setPeriod(100) # period
app.add(zm)

sort_method = galamost.Sort(all_info) # sorting memory to improve performance
sort_method.setPeriod(200)# (period)
app.add(sort_method)

xml = galamost.XmlDump(all_info, 'particle') # output the configuration files in xml format
xml.setPeriod(100000)# (period)
xml.setOutputImage(True)
xml.setOutputBond(True)
xml.setOutputVelocity(True)
# xml.setOutputDiameter(True)
xml.setOutputType(True)
xml.setOutputBody(True)
xml.setOutputMass(True)
xml.setOutputCris(True)
xml.setOutputInit(True)
app.add(xml)
# groupAb = galamost.ParticleSet(all_info,['G'])
# xml2=galamost.XmlDump(all_info,groupAb,'bondAb' )
# xml2.setPeriod(10)# (period)
# xml2.setOutputPosition(False)
# xml2.setOutputType(False)
# xml2.setOutputImage(False)
# xml2.setOutputBond(True)
# xml2.setOutputVelocity(False)
# xml2.setOutputDiameter(False)
# xml2.setOutputType(False)
# xml2.setOutputBody(False)
# xml2.setOutputMass(False)
# xml2.setOutputCris(False)
# xml2.setOutputInit(False)
# app.add(xml2)

npcoord=np.loadtxt("npnumber.txt").astype('int').tolist()
group = galamost.ParticleSet(all_info, npcoord)
dcd_period=10
dcd = galamost.DcdDump(all_info, group,'particles',True)
dcd.unpbc(True)
dcd.setPeriod(int(dcd_period))
app.add(dcd)

dt=0.0001
app.setDt(dt)
step=100000
step_sum=0
step_sum+=step
app.run(int(step)) # init the system

# app.add(xml)

dt=0.005
app.setDt(dt)
#ready ro run
# step=100000000
step=int(100000000*STEP_K)
# step=1000000
step_sum+=step
app.run(int(step)) # the number of time steps to run
neighbor_list.printStats() # output the information about neighbor_list
np.savetxt('run_step.txt', np.array([step_sum]))
np.savetxt('dt.txt', np.array([dt]))
np.savetxt('dcd_period.txt', np.array([dcd_period]))

#
#
