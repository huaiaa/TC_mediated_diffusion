import numpy as np
import gala_model
from Galamostxmlcreator import GalamostXmlCreator as creator
from optparse import OptionParser
optParser = OptionParser()

optParser.add_option("-d","--dying", action="store", type=str, dest="dying",
                     default='hex', help="dying_type")
optParser.add_option("-g","--graft", action="store", type=int, dest="graft",
                     default=30, help="graft_num")
optParser.add_option("-s","--step", action="store", type=int, dest="step",
                     default=2000, help="run_step")
options,args=optParser.parse_args()
Particle=creator('Particle.xml')
dying=options.dying
graft = options.graft
step=options.step
lx = 50
ly = 50
lz = 50
box = [lx,ly,lz]
Particle.setbox(box)

p=gala_model.particle_from_kinetics_with_ord(dying=dying, n_graft=graft)
P_pos=p.GetPos(step=step, running_info=True)

P_type=p.GetType(type_0='C', type_g='D', type_ord='E', type_d='F')
P_body=p.GetBody(body=0)
P_mass=p.GetMass(m_0=1.0, m_g=1.0, m_d=1.0, m_ord=0.001)  # /float(graft+1)
P_ord_num=p.Get_ord_num()
P_image=p.GetImage()
P_velocity=p.GetVelocity()

P_h_init=p.GetH_init()
p.SetH_cris([1,1,1,0])
P_h_cris=p.GetH_cris()
num_of_particle=P_pos.shape[0]

Particle.add_posdata(P_pos)
Particle.add_typedata(P_type)
Particle.add_bodydata(P_body)
Particle.add_massdata(P_mass)
Particle.add_imagedata(P_image)
Particle.add_velocitydata(P_velocity)
Particle.add_h_initdata(P_h_init)
Particle.add_h_crisdata(P_h_cris)

Particle.write_sample(filename='Particle.xml')
np.savetxt('npnumber.txt',P_ord_num.astype('int'))
np.savetxt('atom_num.txt',np.array([num_of_particle], dtype=int))
radius = np.linalg.norm(P_pos, axis=-1)[1]
np.savetxt('radius.txt', np.array([radius]))
np.savetxt('step.txt', np.array([step]))