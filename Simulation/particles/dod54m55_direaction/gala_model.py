import numpy as np
import math
import pdb
import matplotlib.pyplot as plt

def bond_connect(pos, b_length):
    dis=np.linalg.norm(pos[:,None,:]-pos[None,:,:], axis=-1)
    con=(np.abs(np.triu(dis)-b_length)<1e-5)
    i,j=np.where(con)
    return np.stack((i,j),axis=1)

def bond_connect_bychains(pos, n_apex, chainlength, b_length):
    natoms=pos.shape[0]
    ones=np.ones((chainlength,chainlength))
    block_diagonal=np.kron(np.eye(((natoms-n_apex)//chainlength)), ones)
    exist=np.zeros((natoms, natoms))
    exist[0:(natoms-n_apex), 0:(natoms-n_apex)]+=block_diagonal
    exist[0:natoms, (natoms-n_apex):natoms]=1
    exist=np.triu(exist)
    dis = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    con=(np.abs(dis*exist-b_length)<1e-5)
    i,j=np.where(con)
    return np.stack((i,j),axis=1)
def period_bond_connect(pos, b_length, box_size):
    dis=np.linalg.norm((pos[:, None, :]-pos[None, :, :]+0.5*box_size) % box_size-0.5*box_size, axis=-1)
    con=(np.abs(np.triu(dis)-b_length)<1e-5)
    i,j=np.where(con)
    return np.stack((i,j),axis=1)
def contact_exam(pos1,p0, boxlen,radius=1.0):
    dis = np.linalg.norm((pos1 - p0+0.5*boxlen)%boxlen-0.5*boxlen, axis=-1)
    j = np.array(np.where(dis<radius)).flatten()
    return j
def add_antibody(antibody_num, A_pos, pos, particle_pos,particle_r, boxlen, step=2000, step_len=1e-2):
    random_pos = (np.random.random((antibody_num, 3)) * 2.0 - 1.0) * boxlen / 2.0
    while True:
        dis = np.linalg.norm((random_pos[:, None, :] - particle_pos[None, :, :] + 0.5 * boxlen) % boxlen - 0.5 * boxlen, axis=-1)
        contact=np.where(dis < particle_r)
        j = np.array(contact[0]).flatten()
        if j.shape[0]==0:
            break
        random_pos[j,:]=(np.random.random((j.shape[0], 3)) * 2.0 - 1.0) * boxlen / 2.0

    pos_all=np.concatenate((random_pos,pos), axis=0)
    v = np.zeros((antibody_num,3),dtype=float)
    for i in range(step):
        dd = (random_pos[:, None, :] - pos_all[None, :, :] + 0.5 * boxlen) % boxlen - 0.5 * boxlen
        dis = np.linalg.norm(dd, axis=-1, keepdims=True)
        dis[dis < 1e-2] = 1e-2
        F = (dd / dis ** 3).sum(axis=1)
        #aa=r[0:(self.num-self.num_fixed), :]
        v=v+F*step_len
        random_pos=(random_pos+v+ 0.5 * boxlen) % boxlen - 0.5 * boxlen
        pos_all[0:antibody_num, :]=random_pos
        if i%10==0:
            print('antibody add step{0}'.format(i))
    AN_pos = ((random_pos[:, None, :] + A_pos).reshape(-1, 3)+ 0.5 * boxlen) % boxlen - 0.5 * boxlen
    return AN_pos


def particle_pos_add(pos1,pos2,boxlen,step=2000, step_len=0.1):
    radius=np.linalg.norm(pos2[1]-pos2[0])+1
    for i in range(step):
        p0 = pos2[0]
        contact=contact_exam(pos1,p0, boxlen, radius)
        if contact.shape[0]==0:
            return True, pos1, pos2
        for c in contact:
            p1=pos1[c]
            p2=pos2[0]
            v=(p2-p1+0.5*boxlen)%boxlen-0.5*boxlen
            v=v/np.linalg.norm(v)/np.linalg.norm(v)
            pos2=(pos2+v*step_len+0.5*boxlen)%boxlen-0.5*boxlen
    return False, pos1, pos2
def antibody_pos_add(pos_a, pos0, pos1, pos2, boxlen,step=2000,step_len=0.1):
    radius = np.linalg.norm((pos2[0] - pos2[1])/2.)+0.5
    radius_p = np.linalg.norm(pos0[1] - pos0[0]) + 0.5
    acontact=np.array([])
    for i in range(step):
        p0 = pos2[0]/2. + pos2[1]/2.
        contact=contact_exam(pos1, p0, boxlen, radius)
        for c in contact:
            p1 = pos1[c]
            p2 = p0
            v = (p2-p1+0.5*boxlen)%boxlen-0.5*boxlen
            v = v/np.linalg.norm(v)/np.linalg.norm(v)
            pos2=(pos2+v*step_len+0.5*boxlen)%boxlen-0.5*boxlen
        pcontact = contact_exam(pos0, p0, boxlen, radius+radius_p)
        if contact.shape[0] == 0 and pcontact.shape[0]==0:
            return True, pos2
        for c in pcontact:
            p1 = pos0[c]
            p2 = p0
            v = (p2 - p1 + 0.5 * boxlen) % boxlen - 0.5 * boxlen
            v = v / np.linalg.norm(v) / np.linalg.norm(v)
            pos2 = (pos2 + v * step_len + 0.5 * boxlen) % boxlen - 0.5 * boxlen
        if pos_a.shape[0]>0.1:
            acontact = contact_exam(pos_a, p0, boxlen, radius + radius)
        for c in acontact:
            p1 = pos_a[c]
            p2 = p0
            v = (p2 - p1 + 0.5 * boxlen) % boxlen - 0.5 * boxlen
            v = v / np.linalg.norm(v) / np.linalg.norm(v)
            pos2 = (pos2 + v * step_len + 0.5 * boxlen) % boxlen - 0.5 * boxlen
        if contact.shape[0] == 0 and pcontact.shape[0] == 0 and acontact.shape[0]==0:
            return True, pos2
    return False, pos2

class particle_from_kinetics_with_ord():
    def __init__(self, dying='tet', n_graft=12):
        self.dying=dying
        self.n_graft=n_graft
        self.pos_ord=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]])*0.1
        self.init_set = np.array([0, 0, 0, 0])  # CDEF
        self.cris_set = np.array([1, 1, 1, 1])  # CDEF

    def SetH_init(self, h_init):
        self.init_set = np.array(h_init).flatten()

    def SetH_cris(self, h_cris):
        self.cris_set = np.array(h_cris).flatten()

    def SetDying(self,dying):
        self.dying = dying
    def SetNGraft(self,n_graft):
        self.n_graft = n_graft
    def GetPos(self, step=200, scale=1.01, running_info=False):
        poly = polygon()
        Sphe = spherical_confinement()
        self.fixed = poly.GetApex(self.dying)
        r0, v0 = Sphe.GetPoint(self.n_graft, self.fixed, step, scale=scale, running_info=running_info)
        pos0 = np.array([0.0, 0, 0])
        self.pos = np.concatenate((pos0[None, :], r0, self.pos_ord), axis=0)
        return self.pos

    def Get_ord_num(self):
        return np.arange(self.pos.shape[0]-7,self.pos.shape[0]-3)
    def GetType(self, type_0='C', type_g='D', type_ord='E', type_d='F'):
        type0 = np.array([type_0])
        typeg = np.array([type_g for i in range((self.pos.shape[0]-1-self.fixed.shape[0]-self.pos_ord.shape[0]))])
        typed = np.array([type_d for i in range((self.fixed.shape[0]))])
        typeord=np.array([type_ord for i in range((self.pos_ord.shape[0]))])
        self.type = np.concatenate((type0, typeg, typed, typeord), axis=0)
        type_id0 = np.array([0])
        type_idg = np.array([1 for i in range((self.pos.shape[0] - 1 - self.fixed.shape[0] - self.pos_ord.shape[0]))])
        type_idord = np.array([2 for i in range((self.pos_ord.shape[0]))])
        type_idd = np.array([3 for i in range((self.fixed.shape[0]))])
        self.type_id = np.concatenate((type_id0, type_idg, type_idd, type_idord), axis=0)
        return self.type
    def GetH_init(self):
        return self.init_set[self.type_id]

    def GetH_cris(self):
        return self.cris_set[self.type_id]
    def GetBody(self, body):
        return np.array([body for i in range(self.pos.shape[0])])

    def GetMass(self, m_0=1.0,m_g=1.0,m_d=1.0,m_ord=0.001):
        mass0=np.array([m_0])
        massg=np.array([m_g for i in range((self.pos.shape[0]-1-self.fixed.shape[0]-self.pos_ord.shape[0]))])
        massd = np.array([m_d for i in range((self.fixed.shape[0]))])
        massord = np.array([m_ord for i in range((self.pos_ord.shape[0]))])
        mass = np.concatenate((mass0, massg, massd, massord), axis=0)
        return mass
    def GetImage(self):
        image=np.zeros((self.pos.shape[0],3),dtype=int)
        return image
    def GetVelocity(self):
        velocity=np.zeros((self.pos.shape[0],3),dtype=float)
        return velocity



class period_spatial_net():
    def __init__(self,size=5, chainlen = 20):
        self.size = size
        self.chainlen = chainlen
        self.init_set = np.array([0, 0]) #A,B
        self.cris_set = np.array([1, 1]) #A,B

    def SetH_init(self, h_init):
        self.init_set = np.array(h_init).flatten()

    def SetH_cris(self, h_cris):
        self.cris_set = np.array(h_cris).flatten()
    def SetSize(self,size):
        self.size = size
    def SetChainlen(self, chainlen):
        self.chainlen = chainlen
    def SetType(self, type_set, chaintype, gridtype):
        self.type_set = type_set
        self.chaintype = chaintype
        self.gridtype = gridtype
    def GetGrid(self):
        num_atom=self.pos.shape[0]
        num_grid=self.grid_real.shape[0]
        return np.arange(num_atom-num_grid, num_atom, dtype=int)
    def GetGridConnect(self):
        return self.grid_connect

    def GetPos(self):
        self.grid_num_all = (self.size + 1) * (self.size + 1) * (self.size + 1)
        self.grid_num_real = self.size * self.size * self.size

        dl = np.arange(0, self.size)
        ones = np.ones(self.size)
        self.grid_real = np.zeros((self.grid_num_real, 3))
        self.grid_real[0:self.grid_num_real, 0] = np.kron(np.kron(dl, ones), ones)
        self.grid_real[0:self.grid_num_real, 1] = np.kron(ones, np.kron(dl, ones))
        self.grid_real[0:self.grid_num_real, 2] = np.kron(ones, np.kron(ones, dl))

        dl = np.arange(0, self.size + 1)
        ones = np.ones(self.size + 1)
        self.grid_all = np.zeros((self.grid_num_all, 3))
        self.grid_all[:, 0] = np.kron(np.kron(dl, ones), ones)
        self.grid_all[:, 1] = np.kron(ones, np.kron(dl, ones))
        self.grid_all[:, 2] = np.kron(ones, np.kron(ones, dl))

        bond = bond_connect(self.grid_all, 1)
        self.grid_connect = bond_connect(self.grid_real, 1)
        self.grid_real *= self.chainlen
        self.grid_all *= self.chainlen
        pi = self.grid_all[bond[:, 0]]
        qi = self.grid_all[bond[:, 1]]
        chains = qi[:, None] + (pi[:, None] - qi[:, None]) / self.chainlen * np.arange(1, self.chainlen)[:, None]
        chains = chains.reshape((-1, 3))
        con = chains.max(axis=1) < self.size * self.chainlen
        i = np.where(con)
        chains = chains[i, :].reshape(-1, 3)
        self.pos = np.concatenate((chains, self.grid_real), axis=0)
        self.pos = self.pos - np.array(
            [self.size * self.chainlen, self.size * self.chainlen, self.size * self.chainlen]) / 2
        return self.pos
    def GetType(self):
        chain_n = (self.pos.shape[0] - self.grid_real.shape[0]) // (self.chainlen - 1)
        self.type_id = np.zeros((chain_n, (self.chainlen - 1)), dtype=np.int)
        self.type_id = self.type_id + self.chaintype
        self.type_id = self.type_id.flatten()
        grid_type_id = np.zeros(self.grid_real.shape[0], dtype=np.int) + self.gridtype
        self.type_id = np.concatenate((self.type_id, grid_type_id), axis=0)
        self.type = self.type_set[self.type_id[:, ]]
        return self.type

    def GetH_init(self):
        return self.init_set[self.type_id[:, ]]

    def GetH_cris(self):
        return self.cris_set[self.type_id[:, ]]

    def GetBond(self):
        bond = period_bond_connect(self.pos, 1, self.size*self.chainlen)
       # bond = bond_connect(self.pos, 1)
        bondtype = self.type[bond[:, 0]] + '-' + self.type[bond[:, 1]]
        self.bonddata = np.empty((len(bond), 3), '<U8')
        self.bonddata[:, 0] = bondtype
        self.bonddata[:, 1:3] = bond
        return self.bonddata
    def GetBody(self, body):
        return np.array([body for i in range(self.pos.shape[0])])
    def GetMass(self, mass=1.0):
        return np.array([mass for i in range(self.pos.shape[0])])

class polygon():
    def __init__(self):
        self.apex={}
        #ico
        m = math.sqrt(50 - 10 * math.sqrt(5)) / 10
        n = math.sqrt(50 + 10 * math.sqrt(5)) / 10
        re = np.array([1, -1])
        ones = np.ones(2)
        reverse = np.zeros((8, 3))
        reverse[:, 0] = np.kron(np.kron(re, ones), ones)
        reverse[:, 1] = np.kron(ones, np.kron(re, ones))
        reverse[:, 2] = np.kron(ones, np.kron(ones, re))
        a1 = np.array([0, n, m]) * reverse[[0, 1, 2, 3]]
        a2 = np.array([m, 0, n]) * reverse[[0, 1, 4, 5]]
        a3 = np.array([n, m, 0]) * reverse[[0, 2, 4, 6]]
        apex = np.concatenate((a1, a2, a3), axis=0)
        apex = apex / np.linalg.norm(apex, ord=2, axis=1)[:, None]
        self.apex['ico']=apex
        #oct
        apex=np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        apex = apex / np.linalg.norm(apex, ord=2, axis=1)[:, None]
        self.apex['oct']=apex
        # tet
        apex = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1, 0, 1], [0, 1, 1]])-np.array([2,2,2])/4
        apex = apex / np.linalg.norm(apex, ord=2, axis=1)[:, None]
        self.apex['tet'] = apex
        # hex
        apex = np.array([[0, 0, 0], [1., 0, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1]])-np.array([4,4,4])/8
        apex = apex / np.linalg.norm(apex, ord=2, axis=1)[:, None]
        self.apex['hex'] = apex
        # dod
        phi = (math.sqrt(5) + 1) / 2
        re = np.array([1, -1])
        ones = np.ones(2)
        reverse = np.zeros((8, 3))
        reverse[:, 0] = np.kron(np.kron(re, ones), ones)
        reverse[:, 1] = np.kron(ones, np.kron(re, ones))
        reverse[:, 2] = np.kron(ones, np.kron(ones, re))
        a1 = np.array([0, phi, 1 / phi]) * reverse[[0, 1, 2, 3]]
        a2 = np.array([1 / phi, 0, phi]) * reverse[[0, 1, 4, 5]]
        a3 = np.array([phi, 1 / phi, 0]) * reverse[[0, 2, 4, 6]]
        a4 = np.array([1, 1, 1]) * reverse
        apex = np.concatenate((a1, a2, a3, a4), axis=0)
        apex = apex / np.linalg.norm(apex, ord=2, axis=1)[:, None]
        self.apex['dod'] = apex

    def GetApex(self,poly_type):

        return self.apex[poly_type]

class spherical_confinement():
    def __init__(self,G=1e-2):
        self.G=G
    def SetG(self,G):
        self.G = G
    def CountNext(self,r,v):
        dd = r[:, None, :] - r[None, :, :]
        dis = np.linalg.norm(dd, axis=-1, keepdims=True)
        dis[dis < 1e-2] = 1e-2
        F = (dd / dis ** 3).sum(axis=1)
        Fr = (F * r).sum(-1, keepdims=True) * r
        Fv = F - Fr
        #aa=r[0:(self.num-self.num_fixed), :]
        r[0:(self.num-self.num_fixed), :] = r[0:(self.num-self.num_fixed), :] + v[0:(self.num-self.num_fixed), :]
        r = r / np.linalg.norm(r, axis=-1, keepdims=True)
        v = v + self.G * Fv
        return r, v

    def GetPoint(self, num, fixed_points, step, scale=1.01, running_info=False):
        self.num=num
        if fixed_points is not None:
            self.num_fixed = fixed_points.shape[0]
        else:
            self.num_fixed = 0
        a = np.random.random((self.num-self.num_fixed, 1)) * 2 * np.pi  # 根据随机求面均匀分布，先生成一个初始状态
        b = np.arcsin(np.random.random((self.num-self.num_fixed, 1)) * 2 - 1)
        r0 = np.concatenate((np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)), axis=-1)
        if fixed_points is not None:
            r0=np.concatenate((r0,fixed_points),axis=0)
        v0 = np.zeros(r0.shape)
        for ii in range(step):  # 模拟200步，一般已经收敛，其实可以在之下退出
            [rn, vn] = self.CountNext(r0, v0)  # 更新状态
            r0 = rn
            v0 = vn
        dd = r0[:, None, :] - r0[None, :, :]
        dis = np.linalg.norm(dd, axis=-1, keepdims=True).reshape(self.num, self.num)
        index = np.triu_indices(self.num, 1)
        dis = dis[index]
        r0=r0/np.min(dis)*scale
        if running_info:
            dd = r0[:, None, :] - r0[None, :, :]
            dis = np.linalg.norm(dd, axis=-1, keepdims=True).reshape(self.num, self.num)
            index = np.triu_indices(self.num, 1)
            dis = dis[index]
            R = np.linalg.norm(r0, axis=-1, keepdims=True)
            info={}
            info['meanDis'] = np.mean(dis)
            info['varDis'] = np.var(dis)
            info['maxDis'] = np.max(dis)
            info['scale'] = scale
            info['radius'] = float(R[0])
            print('mean: {:16.10f}, variance: {:16.10f}, '
                  'max: {:16.10f}, scale: {:16.10f},'
                  ' radius: {:16.10f}'.format(info['meanDis'], info['varDis'], info['maxDis'], info['scale'],info['radius']))
            np_info=np.array([(key, info[key]) for key in info.keys()])
            np.savetxt('particleInfo.txt', np_info, fmt = '%s')
            np.savetxt('particleInfoPure.txt', np.array([info[key] for key in info.keys()]))
        return r0, v0

class antibody():
    def __init__(self, angle1=140.0, angle2=140.0):
        self.angle1=angle1
        self.angle2=angle2


    def SetAngle(self,angle1=140.0, angle2=140.0):
        self.angle1 = angle1
        self.angle2 = angle2

    def GetPos(self):
        pos0=np.array([0.0,0.0,0.0])
        posFc=np.array([0.0,0.0,-1.0])
        posFab1=np.array([np.cos((self.angle1-90.0)/360.0*2.0*np.pi), 0.0, np.sin((self.angle1-90.0)/360.0*2.0*np.pi)])
        posFab2=np.array([-np.cos((self.angle2-90.0)/360.0*2.0*np.pi), 0.0, np.sin((self.angle2-90.0)/360.0*2.0*np.pi)])

       # print(np.sin((self.angle2-90.0)/360.0*2.0*np.pi))
        self.pos = np.concatenate((pos0,posFc,posFab1,posFab2), axis=0).reshape(-1,3)
        return self.pos

    def GetType(self,  type_0='K',type_fab='H', type_fc='J'):
        type0 = np.array([type_0])
        typeFc = np.array([type_fc])
        typeFab = np.array([type_fab,type_fab])
        type = np.concatenate((type0, typeFc, typeFab), axis=0)
        return type

    def GetBody(self, body):
        return np.array([body for i in range(self.pos.shape[0])])
    def GetMass(self, mass):
        # mass0=np.array([m_0])
        # massg=np.array([m_g for i in range((self.pos.shape[0]-1-self.fixed.shape[0]-self.pos_ord.shape[0]))])
        # massd = np.array([m_d for i in range((self.fixed.shape[0]))])
        # massord = np.array([m_ord for i in range((self.pos_ord.shape[0]))])
        # mass = np.concatenate((mass0, massg, massd, massord), axis=0)
        mass = np.array([mass for i in range(self.pos.shape[0])])
        return mass
    def GetImage(self):
        image=np.zeros((self.pos.shape[0],3),dtype=int)
        return image
    def GetVelocity(self):
        velocity=np.zeros((self.pos.shape[0],3),dtype=float)
        return velocity
class antibody_simplified():
    def __init__(self,dis=1.0):
        self.dis=dis
        self.init_set = np.array([0, 0])  # G,H
        self.cris_set = np.array([1, 1])  # G,H
        self.type_id=np.array([0,1])

    def SetH_init(self, h_init):
        self.init_set = np.array(h_init).flatten()

    def SetH_cris(self, h_cris):
        self.cris_set =np.array(h_cris).flatten()

    def SetDis(self,dis=1.0):
        self.dis=dis

    def GetPos(self):
        posFab = np.array([0.0, 0.0, self.dis / 2.0])
        posFc = np.array([0.0, 0.0, -self.dis / 2.0])
       # print(np.sin((self.angle2-90.0)/360.0*2.0*np.pi))
        self.pos = np.concatenate((posFab,posFc), axis=0).reshape(-1,3)
        return self.pos

    def GetType(self,  type_fab='G',type_fc='H'):
        typeFab = np.array([type_fab])
        typeFc = np.array([type_fc])
        type = np.concatenate((typeFab, typeFc), axis=0)
        return type
    def GetH_init(self):
        return self.init_set[self.type_id]

    def GetH_cris(self):
        return self.cris_set[self.type_id]
    def GetBond(self):
        self.bonddata = np.empty((1, 3), '<U8')
        self.bonddata[0, 0] = 'G-H'
        self.bonddata[0, 1] = 0
        self.bonddata[0, 2] = 1
        return self.bonddata
    def GetBody(self, body=-1):
        return np.array([body for i in range(self.pos.shape[0])])
    def GetMass(self, mass=1.0):
        # mass0=np.array([m_0])
        # massg=np.array([m_g for i in range((self.pos.shape[0]-1-self.fixed.shape[0]-self.pos_ord.shape[0]))])
        # massd = np.array([m_d for i in range((self.fixed.shape[0]))])
        # massord = np.array([m_ord for i in range((self.pos_ord.shape[0]))])
        # mass = np.concatenate((mass0, massg, massd, massord), axis=0)
        mass = np.array([mass for i in range(self.pos.shape[0])])
        return mass
    def GetImage(self):
        image=np.zeros((self.pos.shape[0],3),dtype=int)
        return image
    def GetVelocity(self):
        velocity=np.zeros((self.pos.shape[0],3),dtype=float)
        return velocity
class particle_from_ico():
    def __init__(self):
        m = math.sqrt(50 - 10 * math.sqrt(5)) / 10
        n = math.sqrt(50 + 10 * math.sqrt(5)) / 10
        re = np.array([1, -1])
        ones = np.ones(2)
        reverse = np.zeros((8, 3))
        reverse[:, 0] = np.kron(np.kron(re, ones), ones)
        reverse[:, 1] = np.kron(ones, np.kron(re, ones))
        reverse[:, 2] = np.kron(ones, np.kron(ones, re))
        a1 = np.array([0, n, m]) * reverse[[0, 1, 2, 3]]
        a2 = np.array([m, 0, n]) * reverse[[0, 1, 4, 5]]
        a3 = np.array([n, m, 0]) * reverse[[0, 2, 4, 6]]
        self.apex = np.concatenate((a1, a2, a3), axis=0)
        self.plane = np.array(
            [0, 10, 1, 0, 1, 8, 0, 8, 4, 0, 4, 6, 0, 6, 10, 1, 10, 7, 1, 7, 5, 1, 5, 8, 5, 8,
             9, 9, 8, 4, 4, 9, 2, 4, 2,
             6,2, 6, 11, 6, 11, 10, 10, 11, 7, 3, 5, 9, 3, 9, 2, 3, 2, 11, 3, 11, 7, 3, 7, 5])
        self.M = np.array(
            [[1, 0.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0.5, 0.5], [0, 0.5, 0, 1, 0.5, 0.5, 0, 0.5, 0, 0.5, 0, 0.5],
             [0, 0, 0.5, 0, 0, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0]]).T
        self.N = np.array([1 / 3, 1 / 3, 1 / 3])
    def GetPos(self, particle_id, iterator):
        pos = np.array(self.apex[self.plane]).reshape(60, 3)
        pos = pos / np.linalg.norm(pos, ord=2, axis=1)[:, None]
        pos = pos.reshape(20, 3, 3)
        for i in range(iterator):
            pos = np.matmul(self.M, pos).reshape(-1, 3)
            pos = pos / np.linalg.norm(pos, ord=2, axis=1)[:, None]
            pos = pos.reshape(20 * int(math.pow(4, i + 1)), 3, 3)
        if particle_id==1:
            pos = np.matmul(self.N, pos).reshape(-1, 3)
            pos = pos / np.linalg.norm(pos, ord=2, axis=1)[:, None]
        elif particle_id==2:
            pos = pos.reshape(-1, 3)
            pos = np.unique(pos, axis=0)
        pos0 = np.array([0.0, 0, 0])
        pos = np.concatenate((pos0[None, :], pos), axis=0)
        return pos

    def GetType(self, natoms,type_mid, type_around):
        type = np.array([type_around for i in range(natoms)])
        type0 = np.array([type_mid])
        type = np.concatenate((type0, type), axis=0)
        return type

    def GetBody(self, natoms, body):
        return np.array([ body for i in range(natoms)])
class particle_from_oct():
    def __init__(self):
        self.apex = np.array([[1.0,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
        self.plane = np.array([0,3,4,0,2,4,2,1,4,3,1,4,0,3,5,0,2,5,2,1,5,3,1,5])
        self.M = np.array(
            [[1, 0.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0.5, 0.5], [0, 0.5, 0, 1, 0.5, 0.5, 0, 0.5, 0, 0.5, 0, 0.5],
             [0, 0, 0.5, 0, 0, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0]]).T
        self.N = np.array([1 / 3, 1 / 3, 1 / 3])
    def GetPos(self, particle_id, iterator):
        pos = np.array(self.apex[self.plane]).reshape(24, 3)
        pos = pos / np.linalg.norm(pos, ord=2, axis=1)[:, None]
        pos = pos.reshape(8, 3, 3)
        for i in range(iterator):
            pos = np.matmul(self.M, pos).reshape(-1, 3)
            pos = pos / np.linalg.norm(pos, ord=2, axis=1)[:, None]
            pos = pos.reshape(8 * int(math.pow(4, i + 1)), 3, 3)
        if particle_id==1:
            pos = np.matmul(self.N, pos).reshape(-1, 3)
            pos = pos / np.linalg.norm(pos, ord=2, axis=1)[:, None]
        elif particle_id==2:
            pos = pos.reshape(-1, 3)
            pos=np.concatenate((self.apex,pos),axis=0)
            dis = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)+1000
            con = (np.abs(dis-np.tril(dis)-1000) < 1e-5)
            i, j = np.where(con)
            all=np.arange(0,pos.shape[0])
            unique=np.setdiff1d(all, j, assume_unique=False)
            pos=pos[unique]
        pos0 = np.array([0.0, 0, 0])
        pos = np.concatenate((pos0[None, :], pos), axis=0)
        return pos

    def GetType(self, natoms, type_mid, type_around, type_dye):
        type = np.array([type_around for i in range(natoms-self.apex.shape[0])])
        type0 = np.array([type_mid])
        type_d=np.array([type_dye for i in range(self.apex.shape[0])])
        type = np.concatenate((type0,type_d, type), axis=0)
        return type

    def GetBody(self, natoms, body):
        return np.array([ body for i in range(natoms)])
class particle_from_tet():
    def __init__(self):
        self.apex = np.array([[0.0,0.0,0.0], [1.0,1.0,0.0], [1,0,1],[0,1,1]])-np.array([1,1,1])/2
        self.plane = np.array([0,1,2,0,1,3,1,2,3,0,2,3])
        self.M = np.array(
            [[1, 0.5, 0.5, 0, 0.5, 0, 0, 0, 0.5, 0, 0.5, 0.5], [0, 0.5, 0, 1, 0.5, 0.5, 0, 0.5, 0, 0.5, 0, 0.5],
             [0, 0, 0.5, 0, 0, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0]]).T
        self.N = np.array([1 / 3, 1 / 3, 1 / 3])
    def GetPos(self, particle_id, iterator):
        pos = np.array(self.apex[self.plane]).reshape(12, 3)
        pos = pos / np.linalg.norm(pos, ord=2, axis=1)[:, None]
        pos = pos.reshape(4, 3, 3)
        for i in range(iterator):
            pos = np.matmul(self.M, pos).reshape(-1, 3)
            pos = pos / np.linalg.norm(pos, ord=2, axis=1)[:, None]
            pos = pos.reshape(4 * int(math.pow(4, i + 1)), 3, 3)
        if particle_id==1:
            pos = np.matmul(self.N, pos).reshape(-1, 3)
            pos = pos / np.linalg.norm(pos, ord=2, axis=1)[:, None]
        elif particle_id==2:
            pos = pos.reshape(-1, 3)
            pos = np.unique(pos, axis=0)
        pos0 = np.array([0.0, 0, 0])
        pos = np.concatenate((pos0[None, :], pos), axis=0)
        return pos

    def GetType(self, natoms,type_mid, type_around):
        type = np.array([type_around for i in range(natoms)])
        type0 = np.array([type_mid])
        type = np.concatenate((type0, type), axis=0)
        return type

    def GetBody(self, natoms, body):
        return np.array([ body for i in range(natoms)])

class particle_from_kinetics():
    def __init__(self, dying='tet', n_graft=12):
        self.dying=dying
        self.n_graft=n_graft
    def SetDying(self,dying):
        self.dying = dying
    def SetNGraft(self,n_graft):
        self.n_graft = n_graft
    def GetPos(self, step=200, running_info=False):
        poly = polygon()
        Sphe = spherical_confinement()
        self.fixed = poly.GetApex(self.dying)
        r0, v0 = Sphe.GetPoint(self.n_graft, self.fixed, step, running_info)
        pos0 = np.array([0.0, 0, 0])
        self.pos = np.concatenate((pos0[None, :], r0), axis=0)
        return self.pos

    def GetType(self, type_0, type_g, type_d):
        type0 = np.array([type_0])
        typeg = np.array([type_g for i in range((self.pos.shape[0]-1-self.fixed.shape[0]))])
        typed = np.array([type_d for i in range((self.fixed.shape[0]))])
        type = np.concatenate((type0, typeg, typed), axis=0)
        return type

    def GetBody(self, body):
        return np.array([body for i in range(self.pos.shape[0])])
class spatial_net():
    def __init__(self,size=5, chainlen = 20):
       self.size=size
       self.chainlen=chainlen
    def SetSize(self,size):
        self.size = size
    def SetChainlen(self, chainlen):
        self.chainlen = chainlen
    def SetType(self, type_set, chaintype, gridtype):
        self.type_set = type_set
        self.chaintype = chaintype
        self.gridtype = gridtype

    def GetPos(self):
        dl = np.arange(0, self.size + 1)
        ones = np.ones(self.size + 1)
        self.grid = np.zeros(((self.size + 1) * (self.size + 1) * (self.size + 1), 3))
        self.grid[:, 0] = np.kron(np.kron(dl, ones), ones)
        self.grid[:, 1] = np.kron(ones, np.kron(dl, ones))
        self.grid[:, 2] = np.kron(ones, np.kron(ones, dl))
        self.grid=self.grid-np.array([self.size,self.size,self.size])/2
        bond = bond_connect(self.grid, 1)
        self.grid *= self.chainlen
        pi = self.grid[bond[:, 0]]
        qi = self.grid[bond[:, 1]]
        chains = qi[:, None] + (pi[:, None] - qi[:, None]) / self.chainlen * np.arange(1, self.chainlen)[:, None]
        chains = chains.reshape((-1, 3))
        self.pos = np.concatenate((chains, self.grid), axis=0)
        return self.pos
    def GetType(self):
        chain_n = (self.pos.shape[0] - self.grid.shape[0]) // (self.chainlen - 1)
        type_id = np.zeros((chain_n, (self.chainlen - 1)), dtype=np.int)
        type_id = type_id + self.chaintype
        type_id = type_id.flatten()
        grid_type_id = np.zeros(self.grid.shape[0], dtype=np.int) + self.gridtype
        type_id = np.concatenate((type_id, grid_type_id), axis=0)
        self.type = self.type_set[type_id[:, ]]
        return self.type
    def GetBond(self):
        bond = bond_connect(self.pos, 1)
        bondtype = self.type[bond[:, 0]] + '-' + self.type[bond[:, 1]]
        self.bonddata = np.empty((len(bond), 3), '<U8')
        self.bonddata[:, 0] = bondtype
        self.bonddata[:, 1:3] = bond
        return self.bonddata
    def GetBody(self, body):
        return np.array([body for i in range(self.pos.shape[0])])


if __name__ == '__main__':
    pass
    A=antibody()
    A.GetPos()
    A.GetType()
    # poly=polygon()
    # Sphe=spherical_confinement()
    # fixed=poly.GetApex('hex')
    # step=200
    # radius=np.zeros((51,2))
    # varsD=np.zeros(51)
    # meanD = np.zeros(51)
    # maxD=np.zeros(51)
    # for Ni in range(9,60):
    #     r0, v0 = Sphe.GetPoint(Ni, fixed, step)
    #     R=np.linalg.norm(r0, axis=-1, keepdims=True)
    #     radius[Ni - 9][0] = Ni
    #     radius[Ni-9][1]=R[0]
    #
    #     dd = r0[:, None, :] - r0[None, :, :]
    #     dis = np.linalg.norm(dd, axis=-1, keepdims=True).reshape(Ni, Ni)
    #     index = np.triu_indices(Ni, 1)
    #     dis = dis[index]
    #     varsD[Ni-9]=np.var(dis)
    #     meanD[Ni-9]=np.mean(dis)
    #     maxD[Ni-9]=np.max(dis)
    #
    # np.savetxt('radius.txt', radius)
    # plt.plot(radius[:,0],radius[:,1], label='radius')
    # plt.plot(radius[:, 0], varsD,label='varsD')
    # plt.plot(radius[:, 0],meanD,label='meanD')
    # plt.plot(radius[:, 0], maxD,label='maxD')
    # plt.legend()
    # plt.xlabel("Number of total graft")
    # plt.show()



    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(r0[:, 0], r0[:, 1], r0[:, 2], s=20)
    # plt.show()





