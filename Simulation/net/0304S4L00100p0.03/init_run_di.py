#!/usr/bin/python
import sys
import numpy as np

sys.path.append('/opt/galamost4/lib')  # the path where the GALAMOST program is installed
import galamost
from optparse import OptionParser

global _options
parser = OptionParser()
parser.add_option('--gpu', dest='gpu', help='GPU on which to execute')
parser.add_option("-p","--phi", action="store", type=float, dest="phi",
                     default=0.04, help="volume_fraction_of_net")
parser.add_option("-n","--netname", action="store", type=str, dest="netname",
                     default='size3ligand32323', help="name_of_net")
(_options, args) = parser.parse_args()

r = 2.**(1./6)/2.
phi=_options.phi
netname=_options.netname
num_net=int(np.loadtxt("net_num.txt").astype('int').tolist())
boxlen_dest=r*((4./3*np.pi*num_net/phi)**(1./3))

filename = 'Net.xml' # initial configuration file
build_method = galamost.XmlReader(filename)
perform_config = galamost.PerformConfig(_options.gpu) # GPU index
all_info = galamost.AllInfo(build_method, perform_config)

dt = 0.005
app = galamost.Application(all_info, dt)

neighbor_list = galamost.NeighborList(all_info, 3.0 ,0.1)#(,rcut, rbuffer)
neighbor_list.addExclusionsFromBodys() # remove the interactions of particles in a same body

lj = galamost.LjForce(all_info, neighbor_list,3.0)
lj.setParams('A', 'A' ,1.0 ,1.0 ,1.0, 1.12246) #type, type, epsilon, sigma, alpha, rcut
lj.setParams('A', 'B' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
lj.setParams('B', 'B' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
# lj.setParams('A', 'C' ,1.0 ,1.0 ,1.0, 1.12246) #type, type, epsilon, sigma, alpha, rcut
# lj.setParams('B', 'C' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
# lj.setParams('C', 'C' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
# lj.setParams('A', 'D' ,1.0 ,1.0 ,1.0, 1.12246) #type, type, epsilon, sigma, alpha, rcut
# lj.setParams('B', 'D' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
# lj.setParams('C', 'D' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
# lj.setParams('D', 'D' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
# lj.setParams('A', 'F' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
# lj.setParams('B', 'F' ,1.0 ,1.0 ,1.0, 2.5)#type, type, epsilon, sigma, alpha, rcut
# lj.setParams('C', 'F' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
# lj.setParams('D', 'F' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut
# lj.setParams('F', 'F' ,1.0 ,1.0 ,1.0, 1.12246)#type, type, epsilon, sigma, alpha, rcut

lj.setEnergy_shift()#The potential is shifted to make the energy be zero at cutoff distance
app.add(lj)

bondforcefene = galamost.BondForceFene(all_info)
bondforcefene.setParams('A-A', 30, 1.5)
bondforcefene.setParams('A-B', 30, 1.5)
bondforcefene.setParams('B-A', 30, 1.5)
# bondforcefene.setParams('B-B', 30, 1.5)
app.add(bondforcefene)


# Brownian dynamics for rigid bodies
# bgroup = galamost.ParticleSet(all_info, 'body') # the group of the body particles
# bdnvt_rigid = galamost.BdNvtRigid(all_info, bgroup, 1.0, 123) # temperature, seed for random number generator
# app.add(bdnvt_rigid)

nbgroup = galamost.ParticleSet(all_info, 'non_body')
bdnvt = galamost.BdNvt(all_info, nbgroup, 1.0, 123)
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
#
# npcoord=np.loadtxt("npnumber.txt").astype('int').tolist()
# group = galamost.ParticleSet(all_info, npcoord)
#
# dcd_period=10
# dcd = galamost.DcdDump(all_info, group,'particles',True)
# dcd.unpbc(True)
# dcd.setPeriod(int(dcd_period))
# app.add(dcd)

xml = galamost.XmlDump(all_info, 'particle') # output the configuration files in xml format
xml.setPeriod(100000)# (period)
xml.setOutputImage(True)
xml.setOutputBond(True)
xml.setOutputVelocity(True)
#xml.setOutputDiameter(True)
xml.setOutputType(True)
xml.setOutputBody(True)
xml.setOutputMass(True)
xml.setOutputCris(True)
xml.setOutputInit(True)
app.add(xml)

time_step = 0

boxlen_init=np.loadtxt('boxlen.txt').astype('float').tolist()[0]
app.setDt(0.00002)
step = 100000
time_step += step
app.run(int(step)) # initial the system
box_len=boxlen_init
balence_step=1000000
group=galamost.ParticleSet(all_info,'all')

boxlen_adjust_step=int(np.abs(boxlen_dest - boxlen_init) / 0.2 * 1000)
axs = galamost.AxialStretching(all_info, group)
v = galamost.VariantLinear()
v.setPoint(time_step, box_len)  # time step, box length.
v.setPoint(time_step + boxlen_adjust_step, boxlen_dest)
axs.setBoxLength(v, 'X')
app.add(axs)
axs.setBoxLength(v, 'Y')
app.add(axs)
axs.setBoxLength(v, 'Z')
app.add(axs)
dt=0.001
app.setDt(dt)
app.run(int(boxlen_adjust_step))
dt=0.005
app.setDt(dt)
app.run(int(balence_step))
time_step= time_step + boxlen_adjust_step + balence_step

box_len=boxlen_dest
# dt=0.005
# app.setDt(dt)
# #ready ro run
# step=1000000
# app.run(int(step)) # the number of time steps to run
neighbor_list.printStats() # output the information about neighbor_list
np.savetxt('init_step.txt', np.array([time_step//100000*100000]))
np.savetxt('boxlen.txt', np.array([box_len]))
#
#
