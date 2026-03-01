import numpy as np
from Galamostxmlreader import GalamostXmlreader as GxmlReader
from MDAnalysis.lib.formats.libdcd import DCDFile
import matplotlib.pyplot as plt
import math
import scipy.io as sio
import torch
import matplotlib.cm as cm
from scipy import interpolate
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.ticker as ticker
import xml.etree.ElementTree as ET
import numpy as np
from io import StringIO
import bz2
from optparse import OptionParser
def fmt1(x,pos):   # 设置colorbar的刻度值
    if x==0:
        return 0
    return str(int((x*100*10*0.005)/100000))+r'$×10^5$'+r'$\tau $'

def AMP_Bond_cal(e_time=20100000,re=False):
    init_file = 'PIN_init.xml'
    PIN = GxmlReader(init_file)
    bond_data0 = PIN.bonddata.astype(str)
    init_bond_num = bond_data0.shape[0]
    xmlperiod = 100000
    start_time = 100000
    end_time = e_time
    end_time=int(np.loadtxt('run_step.txt').astype(int))
    time = np.arange(start_time, end_time + 1, xmlperiod)
    ii = 0
    bond_numAP = np.zeros((end_time - start_time) // xmlperiod + 1, dtype=int)
    bond_numMAP = np.zeros((end_time - start_time) // xmlperiod + 1, dtype=int)
    bond_numMA = np.zeros((end_time - start_time) // xmlperiod + 1, dtype=int)
    for ts in time:
        filename = 'particle.{:0>10d}.xml'.format(ts)
        # PIN = GxmlReader(filename)
        xmldoc = ET.parse(filename)
        configuration = xmldoc.find("./configuration")
        bond = configuration.find('./bond')
        bondfile = StringIO()
        bondfile.write(bond.text)
        bondfile.seek(0)
        bond_data = np.loadtxt(bondfile, dtype='<U8')
        bond_num = bond_data.shape[0]
        antibody_num=int(np.loadtxt("antibody_num.txt").astype('int'))
        jj=bond_num-antibody_num-1
        bondi=bond_data[jj]
        bond_AP=[] #NO.H
        while (bondi[0][0]=='F'):
            bond_AP+=[int(bondi[2])+1]
            jj-=1
            bondi=bond_data[jj]
            pass
        # bn = bond.attrib['num']
        # atom_num=int(configuration.attrib['natoms'])
        # vel=configuration.find('./velocity')
        # vel_data=[]
        # vel_t=vel.text[(5+1+(atom_num-2)*52+0)::]
        # vel_t=np.array(vel_t.split()).astype(float).reshape(-1, 3)
        # e=np.linalg.norm(vel_t, axis=-1)
        # energy=e*e
        # E=energy.sum()
        bond_id2=(bond_data[:,2].astype(int))[:(bond_num-antibody_num)]
        MAP=np.isin(bond_AP, bond_id2)
        bond_numMAP[ii] = np.sum(MAP)
        bond_numAP[ii] = len(bond_AP)
        bond_numMA[ii] = bond_num - init_bond_num - len(bond_AP)
        ii += 1
        if ii % (10) == 0:
            print(ts)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlabel(r'Time step', fontsize=12)
    ax1.set_ylabel(r'Bond Number', fontsize=12)
    dt = 0.005
    ax1.scatter(time, bond_numAP, label='AP', color='red')
    ax1.scatter(time, bond_numMAP, label='MAP', color='black')
    # ax1.scatter(time, bond_numMA, label='MA', color='blue')
    # ax1.scatter(time, bond_numAP, label='AP', color='red')
    # ax1.scatter(time, bond_numMAP, label='MAP', color='black')
    # ax1.tick_params(axis='y')
    # ax2=ax1.twinx()
    # ax2.scatter(time, bond_numMA, label='MA', color='blue')
    # ax2.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc=2)
    # ax2.legend()
    fig1.savefig('BOND_AP.png', dpi=500, bbox_inches='tight')

    ax1.scatter(time, bond_numMA, label='MA', color='blue')
    ax1.legend(loc=2)
    fig1.savefig('BOND_MA.png', dpi=500, bbox_inches='tight')
    nn=(end_time - start_time) // xmlperiod + 1
    np.savetxt('bond_num_MAP.txt', np.array([np.mean(bond_numMA[nn//2::]), np.mean(bond_numAP[nn//2::]),np.mean(bond_numMAP[nn//2::])]))
    sio.savemat('MAP.mat', {'AP': bond_numAP,'MA':bond_numMA,'MAP':bond_numMAP}, do_compression=True)
    pass
def BondDraw(e_time=600000,re=False):
    init_file = 'PIN_init0.xml'
    PIN = GxmlReader(init_file)
    bond_data = PIN.bonddata.astype(str)
    init_bond_num = bond_data.shape[0]
    xmlperiod = 1000
    start_time = 101000
    end_time = e_time
    time = np.arange(start_time, end_time + 1, xmlperiod)
    ii = 0
    bond_num = np.zeros((end_time-start_time)//xmlperiod+1,dtype=int)
    for ts in time:
        filename = 'particle.{:0>10d}.xml'.format(ts)
        # PIN = GxmlReader(filename)
        xmldoc = ET.parse(filename)
        configuration = xmldoc.find("./configuration")
        bond = configuration.find('./bond')
        bn=bond.attrib['num']
        # atom_num=int(configuration.attrib['natoms'])
        # vel=configuration.find('./velocity')
        # vel_data=[]
        # vel_t=vel.text[(5+1+(atom_num-2)*52+0)::]
        # vel_t=np.array(vel_t.split()).astype(float).reshape(-1, 3)
        # e=np.linalg.norm(vel_t, axis=-1)
        # energy=e*e
        # E=energy.sum()
        bond_num[ii] = int(bn)-init_bond_num
        ii += 1
        if ii % (50)==0:
            print(ts)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlabel(r'Time step', fontsize=12)
    ax1.set_ylabel(r'Bond Number', fontsize=12)
    dt=0.005
    ax1.scatter(time, bond_num)
    fig1.savefig('BOND_NUM.png', dpi=500, bbox_inches='tight')
    if re:
        return bond_num
class DCDANALYSE():
    def __init__(self, filename=None):
        self.filename=filename
        self.effS=0.5
        self.effE=0.99

    def SetFilename(self,filename):
        self.filename=filename
    def Initialize(self,K=1.0):
        self.effS = 0.1 / K
        self.a = DCDFile(self.filename)
        xyz = np.array(self.a.readframes(0, 1)[0])
        self.natoms = self.a.header['natoms']
        self.n_frames = self.a.n_frames
        npcoord = np.loadtxt("npnumber.txt").astype('int')
        npcoord = np.array(npcoord).reshape(-1, 4)
        #print(npcoord)
        self.N_particles=npcoord.shape[0]

        self.axes = np.zeros((self.N_particles, self.n_frames, 3, 3))
        self.r_eul = np.zeros((self.N_particles, self.n_frames, 3))
        self.axes[:, 0, 1, 1] = 1
        self.axes[:, 0, 2, 2] = 1
        self.axes[:, 0, 0, 0] = 1
        time = 0
        vv = np.arange(0, self.N_particles*4, 4)
        xx = vv+1
        yy = vv+2
        zz = vv+3
        print(vv)
        for i in range(time, self.n_frames, 10000): #self.n_frames
            xyz = np.array(self.a.readframes(time, time + min(10000, self.n_frames - time))[0])
            xyz = xyz.transpose(1,0,2)
            xcm = xyz[vv, :, :]
            self.r_eul[:, time:time + min(10000, self.n_frames - time), :] = xcm
            self.axes[:, time:time + min(10000, self.n_frames - time), 0, :] = xyz[xx, :, :] - xyz[vv, :, :]
            self.axes[:, time:time + min(10000, self.n_frames - time), 1, :] = xyz[yy, :, :] - xyz[vv, :, :]
            self.axes[:, time:time + min(10000, self.n_frames - time), 2, :] = xyz[zz, :, :] - xyz[vv, :, :]
            time = time + 10000
            print(time)
        # npcoordold = npcoord.copy()
        # npcoord.sort()
        # vv = np.where(npcoord == npcoordold[0])[0][0]
        # xv = np.where(npcoord == npcoordold[1])[0][0]
        # yv = np.where(npcoord == npcoordold[2])[0][0]
        # zv = np.where(npcoord == npcoordold[3])[0][0]
        # vx = vv
        # vy = vv
        # vz = vv
        # axes0 = xyz[0, [xv, yv, zv], :] - xyz[0, [vx, vy, vz], :]
        # if (axes0[0, 0] < 0):
        #     tmp = xv
        #     xv = vx
        #     vx = tmp
        # if (axes0[1, 1] < 0):
        #     tmp = yv
        #     yv = vy
        #     vy = tmp
        # if (axes0[2, 2] < 0):
        #     tmp = zv
        #     zv = vz
        #     vz = tmp
        # time = 0
        # self.axes = np.zeros((self.n_frames, 3, 3))
        # self.r_eul = np.zeros((self.n_frames, 3))
        # self.axes[0, 1, 1] = 1
        # self.axes[0, 2, 2] = 1
        # self.axes[0, 0, 0] = 1
        # for i in range(time, self.n_frames, 10000):
        #     xyz = np.array(self.a.readframes(time, time + min(10000, self.n_frames - time))[0])
        #     xcm = xyz[:, vv, :]
        #     self.r_eul[time:time + min(10000, self.n_frames - time), :] = xcm
        #     self.axes[time:time + min(10000, self.n_frames - time), 0, :] = xyz[:, xv, :] - xyz[:, vx, :]
        #     self.axes[time:time + min(10000, self.n_frames - time), 1, :] = xyz[:, yv, :] - xyz[:, vy, :]
        #     self.axes[time:time + min(10000, self.n_frames - time), 2, :] = xyz[:, zv, :] - xyz[:, vz, :]
        #     time = time + 10000
        #     print(time)
        self.dt = np.loadtxt('dt.txt').astype('float').tolist()
        self.Nspan=np.loadtxt('dcd_period.txt').astype('int').tolist()
    def Part_MSD(self):
        r_eul = torch.from_numpy(self.r_eul).type(torch.Tensor).cuda()
        bond_num = BondDraw(200000000, True)
        xmlperiod = 100000
        bias=1000000-100000
        point=np.array(np.where(bond_num<=532)).flatten()
        for p in point:

            st=(xmlperiod*p-1000000+bias)//10
            lt=(xmlperiod*p+1000000+bias)//10
            if lt >200000000//10:
                lt=200000000//10
            step = torch.logspace(0, math.log10(lt - st - 10), 300)
            step = torch.unique(step.type(torch.IntTensor)).numpy().tolist()
            Nstep = len(step)
            MSD_eul_3d = torch.zeros((self.N_particles, Nstep, 1)).cuda()
            for nn in range(self.N_particles):
                i = 0
                for steps in step:
                    tmp1 = r_eul[nn, st:(lt - steps)]
                    tmp2 = r_eul[nn, (st + steps):lt]
                    r2_xyz = (tmp1 - tmp2) ** 2
                    r2_3d = torch.sum(r2_xyz, axis=1)
                    MSD_eul_3d[nn, i] = torch.mean(r2_3d, 0)
                    i += 1
                fig1 = plt.figure()
                ax1 = fig1.add_subplot()
                ax1.set_xlabel(r'Time $t(\tau)$', fontsize=12)
                ax1.set_ylabel(r'MSD $<\Delta r^2(t)>(\sigma^2)$', fontsize=12)
                MSD =MSD_eul_3d.cpu().numpy().squeeze()
                Nspan = self.Nspan
                dt = self.dt
                step=np.array(step).flatten()
                x = (Nspan * dt * step).flatten()
                y = MSD.flatten()
                ax1.loglog(x, y)
                # ax1.set_ylim(1e-4, 1e4)
                # draw scale
                xs2 = np.logspace(-1, 0, 10)
                ys2 = np.logspace(-1.9, -0.9, 10)
                xs1 = np.logspace(3, 4, 10)
                ys1 = np.logspace(1, 2, 10)
                ax1.loglog(xs2, ys2, color='blue')
                # ax.loglog(xs1, ys1, color='blue')
                ax1.text(10 ** (-0.8), 10 ** (-1.2), r'~$t$')
                # ax.text(10 ** 3.2, 10 ** 1.8, r'~$t$')
                fig1.savefig('MSD_part{0}.png'.format(st*10), dpi=500, bbox_inches='tight')
                fig1.clear()
                #draw trace
                pos = self.r_eul.squeeze()[st:lt]
                # N_particles=pos.shape[0]
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlabel(r'$X(\sigma)$', fontsize=12)
                ax.set_ylabel(r'$Y(\sigma)$', fontsize=12)
                ax.set_zlabel(r'$Z(\sigma)$', fontsize=12)
                # pos=np.array([[100,0,0],[1,1,1],[5,8,100],[3,2,55]])
                PLOT = pos[::10].reshape(-1, 1, 3)
                Nplot = PLOT.shape[0]
                n = np.arange(Nplot)
                segments = np.concatenate([PLOT[:-1], PLOT[1:]], axis=1)
                lc = Line3DCollection(segments, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0, Nplot))
                lc.set_array(n)
                lc.set_linewidth(1)
                ax.add_collection3d(lc, zs=PLOT[:, 0, 2], zdir='z')
                x0 = pos[0][0]
                y0 = pos[0][1]
                z0 = pos[0][2]

                ax_len=2.5*2
                ax.set_xlim(x0-ax_len, x0+ax_len)
                ax.set_ylim(y0-ax_len, y0+ax_len)
                ax.set_zlim(z0-ax_len, z0+ax_len)
                cb = fig.colorbar(lc, ax=ax, format=ticker.FuncFormatter(fmt1))
                cb.ax.locator_params(nbins=6)
                fig.savefig('Trace_part{0}.png'.format(st*10), dpi=500, bbox_inches='tight')
                fig.clear()
        pass
    def Data_Cal(self):
        r_eul = torch.from_numpy(self.r_eul).type(torch.Tensor).cuda()
       # axes = torch.from_numpy(self.axes).type(torch.Tensor).cuda()
        time = r_eul.shape[1]
        st = int(self.effS * time)
        lt = int(self.effE * time)
        step = torch.logspace(0, math.log10(lt - st-10-200000), 500)
        step = torch.unique(step.type(torch.IntTensor)).numpy().tolist()
        Nstep = len(step)
        self.MSD_eul_3d = torch.zeros((self.N_particles, Nstep, 1)).cuda()
        self.MSD2_eul_3d = torch.zeros((self.N_particles, Nstep, 1)).cuda()
        self.MSD_eul_1d = torch.zeros((self.N_particles, Nstep, 1)).cuda()
        self.MSD2_eul_1d = torch.zeros((self.N_particles, Nstep, 1)).cuda()
        for nn in range(self.N_particles):
            i=0
            for steps in step:
                tmp1 = r_eul[nn, st:(lt - steps)]
                tmp2 = r_eul[nn, (st + steps):lt]
                r2_xyz = (tmp1 - tmp2) ** 2
                r2_1d = r2_xyz[:, 0]
                r4_1d = r2_1d ** 2
                self.MSD_eul_1d[nn,i] = torch.mean(r2_1d, 0)
                self.MSD2_eul_1d[nn,i] = torch.mean(r4_1d, 0)
                r2_3d = torch.sum(r2_xyz, axis=1)
                r4_3d = r2_3d ** 2
                self.MSD_eul_3d[nn,i] = torch.mean(r2_3d, 0)
                self.MSD2_eul_3d[nn,i] = torch.mean(r4_3d, 0)
                i += 1
        self.Gsinterval = np.array([8000, 16000, 24000, 48000, 96000,384000])  #,12288000, 18000000, 1536000, 6144000
        self.bin = 0.1 * np.arange(1000)
        self.Gsdata = torch.zeros((self.N_particles, self.Gsinterval.shape[0], self.bin.shape[0], 1)).cuda()
        for nn in range(self.N_particles):
            kk=0
            for steps in self.Gsinterval:
                tmp1 = r_eul[nn, st:(lt - steps)]
                tmp2 = r_eul[nn, (st + steps):lt]
                #print(tmp1.size())
               # print(tmp2.size())
                Rsq = torch.sqrt(torch.sum((tmp1 - tmp2) ** 2, axis=1))
                # temp=torch.histc(Rsq, bins=200, min=0, max=20).unsqueeze(1).cuda()
                # print(temp.size())
                self.Gsdata[nn,kk,:]=torch.histc(Rsq, bins=1000, min=0, max=100).unsqueeze(1).cuda()
                kk = kk + 1
                # if kk == 1:
                #     self.Gsdata = torch.histc(Rsq, bins=200, min=0, max=20).unsqueeze(1).cuda()
                #     print(self.Gsdata.shape)
                # else:
                #     tmp = torch.histc(Rsq, bins=200, min=0, max=20)
                #     self.Gsdata = torch.cat((self.Gsdata, tmp.unsqueeze(1)), 1)

        # self-intermediate scattering function

        k=2*np.array([(3.**(1./2))/3, (3**(1./2))/3, (3**(1./2))/3])
        k=torch.from_numpy(k).type(torch.Tensor).cuda()
        self.FsData = torch.zeros((self.N_particles, Nstep, 1)).cuda()
        for nn in range(self.N_particles):
            i=0
            for steps in step:
                tmp1 = r_eul[nn, st:(lt - steps)]
                tmp2 = r_eul[nn, (st + steps):lt]
                r_delta = tmp2 - tmp1
                self.FsData[nn, i] = torch.mean(torch.cos(torch.sum(k * r_delta, axis=1)), 0)
                i += 1

      #  bin = torch.from_numpy(bin).cuda()
        step = np.array(step).flatten()
        slt=np.array([st,lt])
        dt=np.array([self.dt])
        # r_eul = r_eul.cpu().numpy()
        # axes = axes.cpu().numpy()
        MSD_eul_3d = self.MSD_eul_3d.cpu().numpy()
        MSD2_eul_3d = self.MSD2_eul_3d.cpu().numpy()
        MSD_eul_1d = self.MSD_eul_1d.cpu().numpy()
        MSD2_eul_1d = self.MSD2_eul_1d.cpu().numpy()
       # bin = bin.cpu().numpy()
        Gsdata = self.Gsdata.cpu().numpy()
        FsData=self.FsData.cpu().numpy()
        filesave = './data.mat'
        sio.savemat(filesave, {'slt':slt, 'step': step,
                               'MSD_eul_3d': MSD_eul_3d,'MSD2_eul_3d': MSD2_eul_3d,
                               'MSD_eul_1d': MSD_eul_1d,'MSD2_eul_1d': MSD2_eul_1d,
                               'Gsinterval': self.Gsinterval.flatten(), 'bin': self.bin, 'Gsdata': Gsdata,
                               'FsData': FsData,
                               'dt': dt, 'Nspan': self.Nspan},
                                                              do_compression=True)  # 'r_eul': self.r_eul, 'axes': self.axes,
    def figDrawing(self):
        data = sio.loadmat('data.mat')
        fig1 = plt.figure()
        ax1 = fig1.add_subplot()
        ax1.set_xlabel(r'Time $t(\tau)$', fontsize=12)
        ax1.set_ylabel(r'MSD $<\Delta r^2(t)>(\sigma^2)$', fontsize=12)
        MSD=data['MSD_eul_3d'].squeeze()
        step=data['step']
        Nspan=data['Nspan']
        dt=data['dt']
        x=(Nspan * dt * step).flatten()
        y = MSD.flatten()
        ax1.loglog(x, y, marker=".")
        # ax1.set_ylim(1e-4, 1e4)
        # draw scale
        xs2 = np.logspace(-1, 0, 10)
        ys2 = xs2 * xs2
        xs1 = np.logspace(4, 5, 10)
        ys1 = xs1 * 1e-3
        ax1.loglog(xs2, ys2)
        ax1.loglog(xs1, ys1)
        ax1.text(0.1, 0.2, r'~$t^2$')
        ax1.text(10 ** 4.5, 10 ** 1.3, r'~t')

        fig1.savefig('MSD.png',dpi=500,bbox_inches = 'tight')

        # figdebug = plt.figure()
        # axdebug = figdebug.add_subplot()
        # axdebug.plot(x, yn)
        # figdebug.savefig('debug.png',dpi=500,bbox_inches = 'tight')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot()
        ax2.set_xlabel(r'Time $t(\tau)$', fontsize=12)
        ax2.set_ylabel(r'NGP(3D) $<\alpha>$', fontsize=12)
        MSD = data['MSD_eul_3d'].squeeze()
        MSD2 = data['MSD2_eul_3d'].squeeze()
        NGP = 3 / 5 * MSD2 / MSD ** 2 - 1
        step = data['step']
        Nspan = data['Nspan']
        dt = data['dt']
        x = (Nspan * dt * step).flatten()
        y = NGP.flatten()
        # x_num = x.shape[0]
        # ni = np.unique(np.linspace(0, x_num - 1, 20).astype(int))
        # f2 = interpolate.interp1d(np.log10(x[ni]), y[ni], kind='cubic')
        # xn = x[0:(x_num - 1)]
        # yn = f2(np.log10(xn))
        ax2.semilogx(x, y, marker=".")
        # ax2.semilogx(x, y)
        fig2.savefig('NGP_3D.png',dpi=500,bbox_inches = 'tight')


        fig3 = plt.figure()
        ax3 = fig3.add_subplot()
        ax3.set_xlabel(r'Time $t(\tau)$', fontsize=12)
        ax3.set_ylabel(r'NGP(1D) $<\alpha>$', fontsize=12)
        MSD = data['MSD_eul_1d'].squeeze()
        MSD2 = data['MSD2_eul_1d'].squeeze()
        NGP = 1 / 3 * MSD2 / MSD ** 2 - 1
        step = data['step']
        Nspan = data['Nspan']
        dt = data['dt']
        x = (Nspan * dt * step).flatten()
        y = NGP.flatten()
        # x_num = x.shape[0]
        # ni = np.unique(np.linspace(0, x_num - 1, 20).astype(int))
        # f3 = interpolate.interp1d(np.log10(x[ni]), y[ni], kind='cubic')
        # xn = x[0:(x_num - 1)]
        # yn = f3(np.log10(xn))
        ax3.semilogx(x, y, marker=".")
        # for i in range(MSD.shape[0]):
        #     MSDi =MSD[i, :]
        #     MSD2i = MSD2[i, :]
        #     NGPi = 3 / 5 * MSD2i / MSDi ** 2 - 1
        #     step = data['step']
        #     Nspan = data['Nspan']
        #     dt = data['dt']
        #     x = (Nspan * dt * step).flatten()
        #     y = NGPi.flatten()
        #     # x_num = x.shape[0]
        #     # ni = np.unique(np.linspace(0, x_num - 1, 20).astype(int))
        #     # f3 = interpolate.interp1d(np.log10(x[ni]), y[ni], kind='cubic')
        #     # xn = x[0:(x_num - 1)]
        #     # yn = f3(np.log10(xn))
        #     ax3.semilogx(x, y, color=colorsParticle[i], label='particle{}'.format(i))
        #     # ax3.semilogx(xn, yn, color=colorsParticle[i], label='particle{}'.format(i))
        # ax3.legend()
        # ax3.semilogx(x, y)
        fig3.savefig('NGP_1D.png',dpi=500,bbox_inches = 'tight')
        GsData = data['Gsdata'].squeeze().transpose(1, 0)
        fig4 = plt.figure()
        ax4 = fig4.add_subplot()
        left, bottom, width, height = 0.61, 0.61, 0.25, 0.25
        ax44 = fig4.add_axes([left, bottom, width, height])
        ax44.set_xlabel(r'r', fontsize=12)
        ax44.set_ylabel(r'Gs(r,t)', fontsize=12)
        ax4.set_xlabel(r'r', fontsize=12)
        ax4.set_ylabel(r'$4\pi r^2$Gs(r,t)', fontsize=12)
        # a=GsData.sum()
        # b = GsData.sum(axis=0)
        # c=GsData.sum(axis=1)
        Gsinterval = data['Gsinterval'].flatten()
        bin = data['bin'].flatten()
        Gs = GsData / GsData.sum(axis=0) / (bin[1] - bin[0])
        colorsGs = cm.jet(np.linspace(0, 1, Gsinterval.shape[0]))
        Nspan = data['Nspan']
        dt = data['dt']
        i = 0
        x = bin
        x_num = x.shape[0]
        for interval in Gsinterval:
            # xx = Gs[:, i]
            # aa = Gs[:, i] * 4 * np.pi * bin * bin
            # bb = interval * dt * Nspan
            y = Gs[:, i]
            # y[y < 1e-8] = 1e-8
            # start = np.argmax(y)
            # j = np.array(np.where(y[start:] <= 1e-8)).flatten()
            # if j.shape[0] == 0:
            #     j = np.array([y.shape[0]]).flatten() - start - 1
            # stop = start + j[0]
            # ni = np.unique(np.linspace(0, stop, 10).astype(int))
            # f4 = interpolate.interp1d(x[ni], y[ni], kind='cubic')
            # xn = x[:stop]
            # yn = np.concatenate((f4(xn), y[stop::]))
            y44 = y / (4 * np.pi * (bin + (bin[1] - bin[0]) / 2) * (bin + (bin[1] - bin[0]) / 2))
            # y44[y44 < 1e-8] = 1e-8
            # start = np.argmax(y)
            # j = np.array(np.where(y[start:] <= 1e-8)).flatten()
            # if j.shape[0] == 0:
            #     j = np.array([y.shape[0]]).flatten() - start - 2
            # stop = start + j[0]
            # ni = np.unique(np.linspace(0, stop, 10).astype(int))
            # f44 = interpolate.interp1d(x[ni], np.log10(y44[ni]), kind='cubic')
            # yn44 = np.concatenate((np.power(10, f44(xn)), y44[stop:]))
            ax4.plot(x, y, color=colorsGs[i],
                     label='t={0}'.format(float(interval * dt * Nspan)), marker=".")
            ax44.semilogy(x, y44, color=colorsGs[i],
                          label='t={0}'.format(float(interval * dt * Nspan)), marker=".")
            i += 1
        ax4.legend(loc=10)
        fig4.savefig('Gs.png', dpi=500, bbox_inches='tight')


        fig5 = plt.figure()
        ax5 = fig5.add_subplot()
        ax5.set_xlabel(r'Time $t(\tau)$', fontsize=12)
        ax5.set_ylabel(r'$F_s (k,t)$', fontsize=12)
        FsData = data['FsData'].squeeze()
        FsData = FsData[:]
        step = data['step']
        Nspan = data['Nspan']
        dt = data['dt']
        x = (Nspan * dt * step).flatten()
        y = FsData.flatten()
        # x_num = x.shape[0]
        # ni = np.unique(np.linspace(0, x_num - 1, 20).astype(int))
        # f5 = interpolate.interp1d(np.log10(x[ni]), y[ni], kind='cubic')
        # xn = x[0:(x_num - 1)]
        # yn = f5(np.log10(xn))
        ax5.semilogx(x, y, marker=".")
        fig5.savefig('Fs.png', dpi=500, bbox_inches='tight')
    def Draw_trace_eff(self):
        # data = sio.loadmat('data.mat')
        # pos=data['r_eul'].squeeze()
        pos = self.r_eul.squeeze()
        st=int(pos.shape[0]*self.effS)
        lt=int(pos.shape[0]*self.effE)
        pos=pos[st:lt,:]
        # N_particles=pos.shape[0]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(r'$X(\sigma)$', fontsize=12)
        ax.set_ylabel(r'$Y(\sigma)$', fontsize=12)
        ax.set_zlabel(r'$Z(\sigma)$', fontsize=12)
        # pos=np.array([[100,0,0],[1,1,1],[5,8,100],[3,2,55]])
        PLOT = pos[::100].reshape(-1, 1, 3)
        Nplot = PLOT.shape[0]
        n = np.arange(Nplot)
        segments = np.concatenate([PLOT[:-1], PLOT[1:]], axis=1)
        lc = Line3DCollection(segments, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0, Nplot))
        lc.set_array(n)
        lc.set_linewidth(1)
        ax.add_collection3d(lc, zs=PLOT[:, 0, 2], zdir='z')
        x0 = pos[0][0]
        y0 = pos[0][1]
        z0 = pos[0][2]

        ax_len=50*2
        ax.set_xlim(x0-ax_len, x0+ax_len)
        ax.set_ylim(y0-ax_len, y0+ax_len)
        ax.set_zlim(z0-ax_len, z0+ax_len)
        cb = fig.colorbar(lc, ax=ax, format=ticker.FuncFormatter(fmt1))
        cb.ax.locator_params(nbins=6)
        fig.savefig('Trace_eff.png', dpi=500, bbox_inches='tight')
    def Draw_trace(self):
        # data = sio.loadmat('data.mat')
        # pos=data['r_eul'].squeeze()
        pos = self.r_eul.squeeze()
        # N_particles=pos.shape[0]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(r'$X(\sigma)$', fontsize=12)
        ax.set_ylabel(r'$Y(\sigma)$', fontsize=12)
        ax.set_zlabel(r'$Z(\sigma)$', fontsize=12)
        # pos=np.array([[100,0,0],[1,1,1],[5,8,100],[3,2,55]])
        PLOT = pos[::100].reshape(-1, 1, 3)
        Nplot = PLOT.shape[0]
        n = np.arange(Nplot)
        segments = np.concatenate([PLOT[:-1], PLOT[1:]], axis=1)
        lc = Line3DCollection(segments, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0, Nplot))
        lc.set_array(n)
        lc.set_linewidth(1)
        ax.add_collection3d(lc, zs=PLOT[:, 0, 2], zdir='z')
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)
        cb=fig.colorbar(lc,ax=ax,format=ticker.FuncFormatter(fmt1))
        cb.ax.locator_params(nbins=6)
        fig.savefig('Trace.png', dpi=500, bbox_inches='tight')

        # for i in range(N_particles):
    def DcdToMat(self):
        filename = './particles.mat'
        sio.savemat(filename, {'data':  self.r_eul.squeeze()}, do_compression=True)


def MSED_Cal(file_name, grid_num_file="gridnumber.txt", grid_connect_file="gridconnect.txt"):
    PIN = GxmlReader(file_name)
    lx = float(PIN.lx)
    pos = PIN.positiondata
    grid_num = np.array(np.loadtxt(grid_num_file).astype('int').tolist())
    grid_pos = pos[grid_num]
    grid_connect = np.array(np.loadtxt(grid_connect_file).astype('int').tolist())
    posi = grid_pos[grid_connect[:, 0]]
    posj = grid_pos[grid_connect[:, 1]]
    dis = np.linalg.norm((posi - posj + 0.5 * lx) % lx - 0.5 * lx, axis=-1)
    MSED = np.mean(dis)
    MSED_var = np.var(dis)
    return MSED, MSED_var

def MSED_Draw():
    xmlperiod=100000
    start_time=0
    end_time=10000000
    time = np.arange(start_time,end_time+1,xmlperiod)
    dt = 0.005
    MSED=np.zeros(int((end_time-start_time)/xmlperiod+1))
    MSED_var = np.zeros(int((end_time-start_time)/xmlperiod+1))
    ii=0
    for ts in time:
        filename='particle.{:0>10d}.xml'.format(ts)
        MSED[ii],MSED_var[ii]=MSED_Cal(filename)
        ii+=1
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlabel(r'Time $t(\tau)$', fontsize=12)
    ax1.set_ylabel(r'MSED $<R^2>$', fontsize=12)
    ax1.plot(dt*time, MSED)
    fig1.savefig('MSED.png', dpi=500, bbox_inches='tight')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_xlabel(r'Time $t(\tau)$', fontsize=12)
    ax2.set_ylabel(r'MSED_var', fontsize=12)
    ax2.plot(dt * time, MSED_var)
    fig2.savefig('MSED_var.png', dpi=500, bbox_inches='tight')

def BondCal():
    init_file='PIN_init.xml'
    PIN = GxmlReader(init_file)
    bond_data=PIN.bonddata.astype(str)
    init_bond_num=bond_data.shape[0]
    xmlperiod=100000
    start_time=1000000
    end_time=10000000
    time = np.arange(start_time,end_time+1,xmlperiod)
    ii=0.0
    bond_num=0
    for ts in time:
        filename='particle.{:0>10d}.xml'.format(ts)
        PIN = GxmlReader(filename)

        bond_data = PIN.bonddata.astype(str)
        bond_num+=bond_data.shape[0]
        ii+=1.0
    bond_num_mean=bond_num/ii-init_bond_num
    np.savetxt('bond_num_mean.txt', np.array([bond_num_mean]))


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--step", action="store", type=float, dest="step",
                      default=1.0, help="step")
    (_options, args) = parser.parse_args()
    step = _options.step
    e_time = int(100000000 * step + 100000)


    a=DCDANALYSE("particles.dcd")

    # data = sio.loadmat('data.mat')
    # st=data['slt'][0][0]
    # lt=data['slt'][0][1]
    # step=data['step']
    # steps=step[0][240]
    # r_eul=data['r_eul']
    # tmp1 = r_eul[st:(lt - steps)]
    # tmp2 = r_eul[(st + steps):lt]
    # r2_xyz = (tmp1 - tmp2) ** 2

    a.Initialize(K=step)
    a.DcdToMat()
    AMP_Bond_cal(e_time=e_time)
    # a.Part_MSD()
    a.Data_Cal()
    a.Draw_trace()
    a.Draw_trace_eff()
    a.figDrawing()


    # MSED_Draw()
    # b = DCDANALYSE("particles.dcd")
    # b.figDrawing()
    # b.Draw_trace()
    # BondCal()
    # data = sio.loadmat('data.mat')
    #
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot()
    # ax1.set_xlabel(r'Time $t(\tau)$', fontsize=12)
    # ax1.set_ylabel(r'MSD $<\Delta r^2(t)>(\sigma^2)$', fontsize=12)
    # MSD = data['MSD_eul']
    # step = data['step'].flatten()
    # Nspan = 10
    # dt = 0.005
    # ax1.loglog(Nspan * dt * step, MSD.sum(axis=1))
    # fig1.savefig('MSD.png',dpi=500,bbox_inches = 'tight')
    #
    # fig4 = plt.figure()
    # ax4 = fig4.add_subplot()
    # left, bottom, width, height = 0.61, 0.61, 0.25, 0.25
    # ax44 = fig4.add_axes([left, bottom, width, height])
    # ax44.set_xlabel(r'r', fontsize=12)
    # ax44.set_ylabel(r'$G_s(r,t)$', fontsize=12)
    # ax4.set_xlabel(r'r', fontsize=12)
    # ax4.set_ylabel(r'$4\pi r^2G_s(r,t)$', fontsize=12)
    # GsData = data['Gsdata']
    #
    # # a=GsData.sum()
    # b=GsData.sum(axis=0)
    # # c=GsData.sum(axis=1)
    # Gsinterval = data['Gsinterval'].flatten()
    # bin = data['bin'].flatten()
    # Gs = GsData / GsData.sum(axis=0)/(bin[1]-bin[0])
    # colorsGs = cm.jet(np.linspace(0, 1, Gsinterval.shape[0]))
    # i = 0
    # for interval in Gsinterval:
    #     xx=Gs[:, i]
    #     aa=Gs[:, i] * 4 * np.pi * bin * bin
    #     bb=interval * dt * Nspan
    #     ax4.plot(bin, Gs[:, i] , color=colorsGs[i],
    #                  label='t={0}'.format(interval * dt * Nspan))
    #     ax44.semilogy(bin, Gs[:, i]/ (4 * np.pi * (bin+(bin[1]-bin[0])/2) * (bin+(bin[1]-bin[0])/2)), color=colorsGs[i], label='t={0}'.format(interval * dt * Nspan))
    #     i += 1
    # ax4.legend(loc=10)
    # fig4.savefig('Gs.png',dpi=500) # ,bbox_inches = 'tight'
    # MSED_Draw()
    pass
