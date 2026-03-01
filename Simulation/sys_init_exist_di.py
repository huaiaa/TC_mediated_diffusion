import numpy as np
import os
import gala_model
from Galamostxmlcreator import GalamostXmlCreator as creator
from Galamostxmlreader import GalamostXmlreader as GxmlReader
from Galamostxmlchanger import GalamostXmlchanger
from optparse import OptionParser

optParser = OptionParser()

optParser.add_option("-s","--size", action="store", type=int, dest="size",
                     default=4, help="net_size")
optParser.add_option("-c","--chainlen", action="store", type=int, dest="chainlen",
                     default=24, help="chain_length")
optParser.add_option("-d","--dying", action="store", type=str, dest="dying",
                     default='dod', help="dying_of_particle")
optParser.add_option("-g","--graft", action="store", type=int, dest="graft",
                     default=54, help="graft_num")
optParser.add_option("-p","--phi", action="store", type=float, dest="phi",
                     default=0.03, help="volume_fraction_of_net")
optParser.add_option("-n","--netname", action="store", type=str, dest="netname",
                     default='0304S4L01010p0.03', help="name_of_net")
optParser.add_option("-a","--antibody_phi", action="store", type=float, dest="antibody_phi",
                     default=0.0015, help="volume_fraction_of_antibody")
optParser.add_option("-f","--force", action="store", type=float, dest="force",
                     default=9.0, help="force_of_attract")
optParser.add_option("-D","--Dp", action="store", type=float, dest="Dp",
                     default=1.02, help="Pr of depoly")
optParser.add_option("-P","--Pp", action="store", type=float, dest="Pp",
                     default=1.2, help="Pr of poly")
optParser.add_option("-A","--Particle", action="store", type=int, dest="Particle",
                     default=1, help="Particle")
optParser.add_option("-k","--kbond", action="store", type=int, dest="kbond",
                     default=220, help="K of bond")
optParser.add_option("-r","--rP", action="store", type=float, dest="rP",
                     default=1.0, help="Pr of reaction AM")
optParser.add_option("-b","--bP", action="store", type=float, dest="bP",
                     default=1.0, help="Pr of broken AM")
optParser.add_option("-m","--mode", action="store", type=str, dest="mode",
                     default='nve', help="nvt or nve")
optParser.add_option("-N","--date", action="store", type=str, dest="date",
                     default='0709', help="date")
optParser.add_option("-i","--is_fix", action="store", type=int, dest="is_fix",
                     default=0, help="is_fix")
optParser.add_option("-S","--sim_name", action="store", type=str, dest="sim_name",
                     default='sim_name', help="sim_name")
options, args=optParser.parse_args()

PIN = creator('PIN.xml')
size = options.size
chainlen = options.chainlen
dying = options.dying
graft = options.graft
phi = options.phi
netname=options.netname
Particle_num=options.Particle
netB = netname[6:12]
antibody_phi=options.antibody_phi
force=options.force
Dp=options.Dp
Pp=options.Pp
kbond=options.kbond
is_fix=options.is_fix
rP=options.rP
bP=options.bP
mode=options.mode
date=options.date
sim_name = options.sim_name
# sim_name='0309reaction_analyse'
P_pos_name = '0416'+dying+'g{0}p{1}Ab{2}m55S{3}{4}__'.format(graft, phi, antibody_phi,size,netB)
if not os.path.exists('simulation/{0}'.format(sim_name)):
    os.system("cd simulation && mkdir {0}".format(sim_name))

Net=GxmlReader('net/{0}/Net_init.xml'.format(netname))

N_pos = Net.positiondata.astype(float)
N_type = Net.typedata.astype(str)
N_bond = Net.bonddata.astype(str)
N_body = Net.bodydata.astype(int)
N_h_init=Net.h_initdata.astype(int)
N_h_cris=Net.h_crisdata.astype(int)
N_mass = Net.massdata.astype(float)
N_image = Net.imagedata.astype(int)
N_velocity = Net.velocitydata.astype(float)*0.0
num_of_net=N_pos.shape[0]
current_atom_num=num_of_net
N_Grid_num=np.array(np.loadtxt('net/{0}/gridnumber.txt'.format(netname)).astype('int').tolist())
N_Grid_connect=np.array(np.loadtxt('net/{0}/gridconnect.txt'.format(netname)).astype('int').tolist())
if is_fix==1:
    N_type[N_Grid_num] = 'R'

lx=float(Net.lx)
ly=float(Net.ly)
lz=float(Net.lz)

box = [lx, ly, lz]
PIN.setbox(box)

r = 2.**(1./6)/2.
antibody_num = int(antibody_phi*lx*ly*lz/(4./3.*np.pi*r*r*r)/2)
antibody_num_atom=antibody_num*2
# Particle=gala_model.particle_from_kinetics_with_ord(dying='hex', n_graft=30)
Particle=GxmlReader('particles/{0}{1}m55_direaction/Particle.xml'.format(dying, graft))

P_pos = Particle.positiondata.astype(float)
# P_pos=(P_pos+0.5*lx/size+0.5*lx)%lx-0.5*lx
P_type = Particle.typedata.astype(str)
P_body = Particle.bodydata.astype(int)
P_h_init=Particle.h_initdata.astype(int)
P_h_cris=Particle.h_crisdata.astype(int)
P_mass = Particle.massdata.astype(float)
P_image = Particle.imagedata.astype(int)
P_velocity = Particle.velocitydata.astype(float)
P_ord_num = np.array(np.loadtxt('particles/{0}{1}m55_direaction/npnumber.txt'.format(dying, graft)).astype('int').tolist())
num_of_particle=P_pos.shape[0]

# Network=gala_model.period_spatial_net(size=size, chainlen=chainlen)
# type_set=np.char.array(['A','B','E'])
# # x=np.arange(3)
# # b=np.arange(3)
# y=np.zeros(chainlen-1,dtype=np.int)
# # y[8 * x[None, :] + b[:, None]]=1
# y[(chainlen-1)//3:((chainlen-1)//3+3)]=1
# chaintype=y
# Network.SetType(type_set=type_set, chaintype=chaintype, gridtype=0)
# N_pos=Network.GetPos()*init_bond_len
# N_type=Network.GetType()
# N_bond=Network.GetBond()
# N_body=Network.GetBody(-1)
# N_mass=Network.GetMass()
# N_Grid_num=Network.GetGrid()
# N_Grid_connect=Network.GetGridConnect()
# num_of_net=N_pos.shape[0]
antibody=gala_model.antibody_simplified()
A_pos=antibody.GetPos()
A_type=antibody.GetType()
A_bond = antibody.GetBond()
A_bond_num = A_bond[:, 1:3].astype('int')
A_body = antibody.GetBody(-1)
A_mass = antibody.GetMass()
antibody.SetH_init([1, 1]) #GH 1 represents initiator; the default value is 0.
antibody.SetH_cris([0, 0]) #GH  0 represents reactive monomer; 1 represents inert monomer; the default value is 0.
A_h_init = antibody.GetH_init()
A_h_cris = antibody.GetH_cris()
A_image = antibody.GetImage()
A_velocity = antibody.GetVelocity()


pos=N_pos
type=N_type
body=N_body
h_init=N_h_init
h_cris=N_h_cris
mass=N_mass
image=N_image
velocity=N_velocity
bond=N_bond

mesh_len=lx/size
# init_particle_pos=np.array([[0.0,0.0,0.0],[2*mesh_len,2*mesh_len,2*mesh_len], [2*mesh_len,2*mesh_len,-2*mesh_len],
#                             [2*mesh_len,-2*mesh_len, 2*mesh_len], [2*mesh_len, -2*mesh_len,-2*mesh_len]]).reshape(-1,3)
init_particle_pos=np.array(np.loadtxt('simulation/{0}/particle_pos.txt'.format(P_pos_name)).astype('float').tolist())
particle_min_dis=[]
ord_num=[]
particle_pos=[]
i=0
ip=init_particle_pos.reshape(-1,3)
Pi_pos = (P_pos + ip + 0.5 * lx) % lx - 0.5 * lx
pos1 = N_pos
p0 = Pi_pos[0]
dis = np.linalg.norm((pos1 - p0 + 0.5 * lx) % lx - 0.5 * lx, axis=-1)
particle_min_dis = particle_min_dis + [np.min(dis)]
particle_pos += [Pi_pos[0]]

pos = np.concatenate((pos, Pi_pos), axis=0)
type = np.concatenate((type, P_type), axis=0)
body = np.concatenate((body, P_body + i), axis=0)
h_init = np.concatenate((h_init, P_h_init), axis=0)
h_cris = np.concatenate((h_cris, P_h_cris), axis=0)
mass = np.concatenate((mass, P_mass), axis=0)
image = np.concatenate((image, P_image), axis=0)
velocity = np.concatenate((velocity, P_velocity), axis=0)
Pi_ord_num = P_ord_num + current_atom_num
ord_num = ord_num + list(Pi_ord_num)
current_atom_num += Pi_pos.shape[0]
i += 1
# for ip in init_particle_pos:
    # Result, N_pos, Pi_pos = gala_model.particle_pos_add(N_pos, P_pos+ip, lx, step=2000, step_len=0.1)
    # Pi_pos = (P_pos + ip + 0.5*lx)%lx-0.5*lx
    # pos1 = N_pos
    # p0 = Pi_pos[0]
    # dis = np.linalg.norm((pos1 - p0 + 0.5 * lx) % lx - 0.5 * lx, axis=-1)
    # particle_min_dis=particle_min_dis+[np.min(dis)]
    # particle_pos+=[Pi_pos[0]]
    #
    # pos = np.concatenate((pos, Pi_pos), axis=0)
    # type = np.concatenate((type, P_type), axis=0)
    # body = np.concatenate((body, P_body+i), axis=0)
    # h_init = np.concatenate((h_init, P_h_init), axis=0)
    # h_cris = np.concatenate((h_cris, P_h_cris), axis=0)
    # mass = np.concatenate((mass, P_mass), axis=0)
    # image = np.concatenate((image, P_image), axis=0)
    # velocity = np.concatenate((velocity, P_velocity), axis=0)
    # Pi_ord_num = P_ord_num + current_atom_num
    # ord_num=ord_num+list(Pi_ord_num)
    # current_atom_num+=Pi_pos.shape[0]
    # i+=1
ord_num=np.array(ord_num)
particle_pos=np.array(particle_pos)
particle_min_dis=np.array(particle_min_dis)
radius=np.linalg.norm(P_pos[1]-P_pos[0])

# antibody_min_dis=[]
antibody_pos=[]
# random_pos=(np.random.random(antibody_num, 3)*2.0-1.0)*lx/2.0
# AN_pos= (random_pos[:, None, :] + A_pos).reshape(-1, 3)
AN_pos=np.array(np.loadtxt('simulation/{0}/antibody_pos.txt'.format(P_pos_name)).astype('float').tolist())
for i in range(antibody_num):
    # while True:
    #     random_pos = (np.random.random(3) * 2.0 - 1.0) * lx / 2.0
    #     Ai_pos =( (random_pos[None, :] + A_pos).reshape(-1, 3)+0.5*lx)%lx-0.5*lx
    #     dis_p = np.linalg.norm((Ai_pos[:, None, :] - particle_pos[None, :, :] + 0.5 * lx) % lx - 0.5 * lx,
    #                          axis = -1)
    #     contact_p = np.where(dis_p < radius)
    #     j_p = np.array(contact_p[0]).flatten()
    #     dis_a=np.linalg.norm((Ai_pos[:, None, :] - pos[None, :, :] + 0.5 * lx) % lx - 0.5 * lx,
    #                          axis=-1)
    #     contact_a = np.where(dis_a < 1.0)
    #     j_a = np.array(contact_a[0]).flatten()
    #     if j_p.shape[0]==0 and j_a.shape[0]==0:
    #         break
    Ai_pos = (AN_pos[i] - A_pos[0] + A_pos + 0.5* lx )%lx-0.5*lx
    antibody_pos += [AN_pos[i]]
    Ai_bond_num = A_bond_num + current_atom_num
    Ai_bond = A_bond
    Ai_bond[:, 1:3] = Ai_bond_num
    pos = np.concatenate((pos, Ai_pos), axis=0)
    type = np.concatenate((type, A_type), axis=0)
    body = np.concatenate((body, A_body), axis=0)
    h_init = np.concatenate((h_init, A_h_init), axis=0)
    h_cris = np.concatenate((h_cris, A_h_cris), axis=0)
    mass = np.concatenate((mass, A_mass), axis=0)
    image = np.concatenate((image, A_image), axis=0)
    velocity = np.concatenate((velocity, A_velocity), axis=0)
    bond = np.concatenate((bond, Ai_bond), axis=0)
    current_atom_num += 2
    if i % 10 == 0:
         print(i)

# for i in range(antibody_num):
#     Ai_pos = (np.random.random(3)*2.0-1.0)*lx/2.0
#     Result, Ai_pos = gala_model.antibody_pos_add(pos_a=np.array(antibody_pos), pos0=particle_pos,pos1=N_pos, pos2=A_pos + Ai_pos, boxlen=lx, step=200, step_len=0.2)
#     pos1 = N_pos
#     pos0 = particle_pos
#     p0 = Ai_pos[0]/2. + Ai_pos[1]/2.
#     antibody_pos+=[p0]
#     dis1 = np.linalg.norm((pos1 - p0 + 0.5 * lx) % lx - 0.5 * lx, axis=-1)
#     dis2 = np.linalg.norm((pos0 - p0 + 0.5 * lx) % lx - 0.5 * lx, axis=-1)
#     antibody_min_dis = antibody_min_dis + [np.min(dis1), np.min(dis2)]
#     Ai_bond_num=A_bond_num+current_atom_num
#     Ai_bond=A_bond
#     Ai_bond[:,1:3] = Ai_bond_num
#     pos = np.concatenate((pos, Ai_pos), axis=0)
#     type = np.concatenate((type, A_type), axis=0)
#     body = np.concatenate((body, A_body), axis=0)
#     h_init = np.concatenate((h_init, A_h_init), axis=0)
#     h_cris = np.concatenate((h_cris, A_h_cris), axis=0)
#     mass = np.concatenate((mass, A_mass), axis=0)
#     image = np.concatenate((image, A_image), axis=0)
#     velocity = np.concatenate((velocity, A_velocity), axis=0)
#     bond = np.concatenate((bond, Ai_bond), axis=0)
#     current_atom_num += 2
#     if i%500==0:
#         print(i)
# antibody_min_dis=np.array(antibody_min_dis)
antibody_pos=np.array(antibody_pos)
np.savetxt('simulation/{0}/'.format(sim_name) + 'particle_pos.txt', particle_pos.reshape(-1,3))
np.savetxt('simulation/{0}/'.format(sim_name) + 'antibody_pos.txt', antibody_pos.reshape(-1,3))
np.savetxt('simulation/{0}/'.format(sim_name) + 'particle_min_dis.txt', particle_min_dis.flatten())
# np.savetxt('simulation/{0}/'.format(sim_name) + 'antibody_min_dis.txt', antibody_min_dis.reshape(-1,2))
# os._exit()

np.savetxt('simulation/{0}/'.format(sim_name) + 'radius.txt', np.array([radius]))



PIN.add_posdata(pos)
PIN.add_typedata(type)
PIN.add_bonddata(bond)
PIN.add_bodydata(body)
PIN.add_h_initdata(h_init)
PIN.add_h_crisdata(h_cris)
PIN.add_massdata(mass)
PIN.add_imagedata(image)
PIN.add_velocitydata(velocity)


PIN.write_sample(filename='simulation/{0}/'.format(sim_name)+'PIN_init.xml')
np.savetxt('simulation/{0}/'.format(sim_name)+'npnumber.txt',ord_num.astype('int'))
np.savetxt('simulation/{0}/'.format(sim_name)+'gridnumber.txt',N_Grid_num.astype('int'))
np.savetxt('simulation/{0}/'.format(sim_name)+'gridconnect.txt',N_Grid_connect.astype('int'))
np.savetxt('simulation/{0}/'.format(sim_name)+'atom_num.txt',np.array([num_of_particle, num_of_net], dtype=int))
np.savetxt('simulation/{0}/'.format(sim_name)+'antibody_num.txt',np.array([antibody_num], dtype=int))
np.savetxt('simulation/{0}/'.format(sim_name)+'particle_num.txt',np.array([particle_pos.shape[0]], dtype=int))
np.savetxt('simulation/{0}/'.format(sim_name)+'boxlen.txt', np.array([lx, ly, lz]))

#
# os.system("cp ./init_run.py simulation/{0}".format(phi))
# os.system("cd simulation/{0} && python2 init_run.py -p {0}".format(phi))
# init_step=int(np.loadtxt('simulation/{0}/init_step.txt'.format(phi)).astype('int').tolist())
# filename='particle.{:0>10d}.xml'.format(init_step)
# init_name='PIN_init.xml'
# os.rename('simulation/{0}/'.format(phi)+filename, 'simulation/{0}/'.format(phi)+init_name)
# PIN = GalamostXmlchanger('simulation/{0}/'.format(phi)+'PIN_init.xml')
# PIN.SetStep(0)
# PIN.write_sample('simulation/{0}/'.format(phi)+'PIN_init.xml')
# os.rename("D:\\demo\\a.txt","D:\\demo\\b.txt")

