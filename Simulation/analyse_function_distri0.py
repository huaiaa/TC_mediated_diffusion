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
from io import StringIO
import bz2
import os
import pywt

def hist_eq_num_per_bin(x, num_per_bin, save=False, filename='hist.mat'):
    x_sort = np.sort(np.array(x))
    num_x = x_sort.shape[0]
    h_x=[]
    h_y=[]
    bin_num=num_x//num_per_bin
    if bin_num == 0:
        return [np.inf], [np.inf]
    for i in range(bin_num-1):
        h_x+=[np.mean(x_sort[i*num_per_bin:(i+1)*num_per_bin])]
        h_y+=[float(num_per_bin)/(x_sort[(i+1)*num_per_bin-1]-x_sort[i*num_per_bin])]
        # if (x_sort[(i+1)*num_per_bin-1]-x_sort[i*num_per_bin])==0:
        #     a=x_sort[(i+1)*num_per_bin-1]
        #     b=x_sort[i*num_per_bin]
        #     c=(i+1)*num_per_bin-1
        #     d=i*num_per_bin
        #     pass
    h_x += [np.mean(x_sort[(bin_num-1) * num_per_bin:num_x])]
    h_y += [float(num_x - (bin_num-1) * num_per_bin) / (x_sort[num_x - 1] - x_sort[(bin_num-1) * num_per_bin])]
    # if bin_num*num_per_bin<num_x:
    #     h_x+=[np.mean(x_sort[bin_num*num_per_bin:num_x])]
    #     h_y += [float(num_x-bin_num*num_per_bin) / (x_sort[num_x - 1] - x_sort[bin_num * num_per_bin])]
    h_y=np.array(h_y)/num_x
    h_x=np.array(h_x)
    if save:
        if not os.path.exists('./hist'):
            os.system("mkdir ./hist")
        sio.savemat(filename, {'h_x': h_x, 'h_y': h_y}, do_compression=True)
    return h_x, h_y
    pass

def time_distri_cal(d=6.0,cutoff_t=1000000,init_part=0.1):
    data = sio.loadmat('particles.mat')['data']
    data=data[int(init_part*data.shape[0])::]
    r0=data[0]
    t0=0
    dt=[]
    t_series=[]
    cutoff=False
    for i in range(data.shape[0]):
        delta_r=np.linalg.norm(data[i]-r0)
        if delta_r>=d:
            r0=data[i]
            dt+=[i-t0]
            t_series+=[i]
            t0=i
            if i>data.shape[0]-cutoff_t:
                cutoff=True
                break
        if i%10000==0:
            print(i)
    if not cutoff:
        dt += [data.shape[0] - t0]
        np.savetxt('not_cutoff{}.txt'.format(d), np.array([d]))
    # plt.figure(figsize=(20, 10))
    # fre_tuple = plt.hist(dt, bins=100, color='steelblue',log=True)
    sio.savemat('dt_distri_{}.mat'.format(d), {'dt': dt}, do_compression=True)
    sio.savemat('t_series_{}.mat'.format(d), {'t_series': t_series}, do_compression=True)
    # plt.scatter(tt,[1]*len(tt))
    # plt.show()
    pass
def time_distri_fig(d=6.0):
    dt = sio.loadmat('dt_distri_{}.mat'.format(d))['dt'].squeeze()
    t_series=sio.loadmat('t_series_{}.mat'.format(d))['t_series'].squeeze()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlabel(r'Time $t(\tau)$', fontsize=12)
    ax1.set_ylabel(r'$p(t)$', fontsize=12)
    for point_num in [5, 10, 20, 40, 80, 200, 500]:
        h_x_eq, h_y_eq = hist_eq_num_per_bin(dt, point_num, True, './hist/hist_eq_num{}_{}.mat'.format(point_num,d))
        ax1.loglog(h_x_eq, h_y_eq, '.',label='bin size={}'.format(point_num))
    xs2 = np.logspace(3, 4, 10)
    ys2 = np.logspace(-5, -7, 10)
    xs1 = np.logspace(4, 5, 10)
    ys1 = np.logspace(2, 3, 10)
    ax1.loglog(xs2, ys2)
    # ax1.loglog(xs1, ys1)
    ax1.text(10 ** (3.5), 10 ** (-5.5), r'~$t^{-2}$')
    # ax1.text(10 ** 4.1, 10 ** 2.5, r'~t')
    ax1.legend()
    fig1.savefig('t_distri_eq_{}.png'.format(d), dpi=500, bbox_inches='tight')
    hist,bins=np.histogram(np.log10(dt),50,density=False)
    bins_len =np.power(10.0,(bins[1::]))-np.power(10.0,(bins[0:(bins.shape[0]-1)]))
    # bins_len=10^(bins[1::]-bins[0:(bins.shape[0]-1)])
    h_y=hist/bins_len/dt.shape[0]
    h_x=0.5*(bins[1::]+bins[0:(bins.shape[0]-1)])
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_xlabel(r'Time $t(\tau)$', fontsize=12)
    ax2.set_ylabel(r'$p(t)$', fontsize=12)
    ax2.semilogy(h_x, h_y, '.')
    fig2.savefig('t_distri_{}.png'.format(d), dpi=500, bbox_inches='tight')
    ax1.loglog(np.power(10.0,h_x), h_y, '.', label='equal bin length')
    ax1.legend()
    fig1.savefig('t_distri_cmp_{}.png'.format(d), dpi=500, bbox_inches='tight')
    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    ax3.set_xlabel(r'Time $t(\tau)$', fontsize=12)
    ax3.set_ylabel(r'$s$', fontsize=12)
    y_dt=np.arange(t_series.shape[0])
    ax3.plot(t_series, y_dt, '.')
    print(dt.shape[0])
    fig3.savefig('t_series_{}.png'.format(d), dpi=500, bbox_inches='tight')
    pass

def time_distri_cal_overlap_numpy(d=6.0,cutoff_t=1000000,init_part=0.1):
    # discard temporarily
    data = sio.loadmat('particles.mat')['data']
    data = data[int(init_part * data.shape[0])::]
    dt=[]
    cutoff = True
    cut_dt=[]
    cut_i=[]
    end_t=data.shape[0]-cutoff_t
    i=np.arange(data.shape[0]-cutoff_t)
    j=1
    while j <= cutoff_t:
        delta_r = np.linalg.norm(data[i] - data[i+j],axis=1)
        jump=delta_r>=d
        # jump=[True, False, False,True]
        jump_index=np.where(jump)[0]
        if len(jump_index)>0:
            i = np.delete(i, jump_index)
            dt+=[j]*jump_index.shape[0]
        if i.shape[0]>0:
            j+=1
        else:
            break
        if j%10==0:
            print([j,i.shape[0]])
    while i.shape[0]>0:
        cut_index=np.where(i+j >= data.shape[0])[0]
        if len(cut_index)>0:
            # dt+=[j]*cut_index.shape[0]
            cut_dt+=[j]*cut_index.shape[0]
            cut_i+=i[cut_index].tolist()
            cutoff=False
            i = np.delete(i, cut_index)
        if i.shape[0] <= 0:
            break
        delta_r = np.linalg.norm(data[i] - data[i + j], axis=1)
        jump = delta_r >= d
        jump_index = np.where(jump)[0]
        if len(jump_index)>0:
            i = np.delete(i, jump_index)
            dt+=[j]*jump_index.shape[0]
        j+=1
        if j%10==0:
            print([j,i.shape[0]])
    if not cutoff:
        sio.savemat('overlap_cut_dt.mat', {'cut_dt': cut_dt,'cut_i': cut_i}, do_compression=True)
        np.savetxt('overlap_not_cutoff{}.txt'.format(d), np.array([d]))
    sio.savemat('overlap_t_distri.mat', {'dt': dt}, do_compression=True)
    pass
def time_distri_cal_overlap(d=6.0,cutoff_t=1000000,init_part=0.1):
    # discard temporarily
    data = sio.loadmat('particles.mat')['data']
    data = data[int(init_part * data.shape[0])::]
    data=torch.from_numpy(data).type(torch.Tensor).cuda()
    dt=[]
    cutoff = True
    cut_dt=[]
    cut_i=[]
    end_t=data.shape[0]-cutoff_t
    i=np.arange(data.shape[0]-cutoff_t)
    i=torch.from_numpy(i).type(torch.Tensor).cuda().long()
    print(data.shape)
    j=1
    d2=d*d
    while j <= cutoff_t:
        # delta_r = np.linalg.norm(data[i] - data[i+j],axis=1)
        delta_r = torch.sum((data[i] - data[i+j]) ** 2, axis=1)
        jump = delta_r >= d2
        # jump=[True, False, False,True]
        jump_index=torch.nonzero(jump,as_tuple = False)
        if jump_index.shape[0] > 0:
            i = i[~jump]
            dt+=[j]*jump_index.shape[0]
        if i.shape[0]>0:
            j+=1
        else:
            break
        if j%1000==0:
            print([j,i.shape[0]])
            if j >= 0.1*cutoff_t and i.shape[0]>=0.7*data.shape[0]:
                print('break')
                np.savetxt('dt_overlap_break{}.txt'.format(d), np.array([d]))
                break

    while i.shape[0]>0:
        if j < cutoff_t:
            # np.savetxt('dt_overlap_break{}.txt'.format(d), np.array([d]))
            break
        if j >= 2*cutoff_t and i.shape[0] >= 0.2*data.shape[0]:
            print('break')
            np.savetxt('dt_overlap_break{}.txt'.format(d), np.array([d]))
            break
        cut=i+j >= data.shape[0]
        cut_index=torch.nonzero(cut,as_tuple=False)
        if cut_index.shape[0]>0:
            # dt+=[j]*cut_index.shape[0]
            cut_dt+=[j]*cut_index.shape[0]
            cut_i+=i[cut_index].tolist()
            cutoff=False
            i = i[~cut]
        if i.shape[0] <= 0:
            break
        delta_r = torch.sum((data[i] - data[i+j]) ** 2, axis=1)
        jump = delta_r >= d2
        jump_index = torch.nonzero(jump, as_tuple=False)
        if jump_index.shape[0] > 0:
            i = i[~jump]
            dt += [j] * jump_index.shape[0]
        j+=1
        if j%1000==0:
            print([j,i.shape[0]])
    if not cutoff:
        sio.savemat('overlap_cut_dt_{}.mat'.format(d), {'cut_dt': cut_dt,'cut_i': cut_i}, do_compression=True)
        np.savetxt('overlap_not_cutoff_{}.txt'.format(d), np.array([d]))
    sio.savemat('overlap_dt_distri_{}.mat'.format(d), {'dt': dt}, do_compression=True)
    pass
def time_distri_fig_overlap(d=6.0):
    dt = sio.loadmat('overlap_dt_distri_{}.mat'.format(d))['dt'].squeeze()
    # t_series=sio.loadmat('t_series_{}.mat'.format(d))['t_series'].squeeze()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlabel(r'Time $t(\tau)$', fontsize=12)
    ax1.set_ylabel(r'$p(t)$', fontsize=12)
    for point_num in [5000,10000,50000,100000]:
        h_x_eq, h_y_eq = hist_eq_num_per_bin(dt, point_num, True, './hist/overlap_hist_eq_num{}_{}.mat'.format(point_num,d))
        ax1.loglog(h_x_eq, h_y_eq, '.',label='bin size={}'.format(point_num))
    xs2 = np.logspace(3, 4, 10)
    ys2 = np.logspace(-5, -7, 10)
    xs1 = np.logspace(4, 5, 10)
    ys1 = np.logspace(2, 3, 10)
    ax1.loglog(xs2, ys2)
    # ax1.loglog(xs1, ys1)
    ax1.text(10 ** (3.5), 10 ** (-5.5), r'~$t^{-2}$')
    # ax1.text(10 ** 4.1, 10 ** 2.5, r'~t')
    ax1.legend()
    fig1.savefig('overlap_t_distri_eq_{}.png'.format(d), dpi=500, bbox_inches='tight')
    hist,bins=np.histogram(np.log10(dt),50,density=False)
    bins_len =np.power(10.0,(bins[1::]))-np.power(10.0,(bins[0:(bins.shape[0]-1)]))
    # bins_len=10^(bins[1::]-bins[0:(bins.shape[0]-1)])
    h_y=hist/bins_len/dt.shape[0]
    h_x=0.5*(bins[1::]+bins[0:(bins.shape[0]-1)])
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_xlabel(r'Time $t(\tau)$', fontsize=12)
    ax2.set_ylabel(r'$p(t)$', fontsize=12)
    ax2.semilogy(h_x, h_y, '.')
    fig2.savefig('overlap_t_distri_{}.png'.format(d), dpi=500, bbox_inches='tight')
    ax1.loglog(np.power(10.0,h_x), h_y, '.', label='equal bin length')
    ax1.legend()
    fig1.savefig('overlap_t_distri_cmp.png'.format(d), dpi=500, bbox_inches='tight')
    if os.path.exists('overlap_cut_dt_{}.mat'.format(d)):
        time_distri_fig_overlap_cut(d)
    pass
def time_distri_fig_overlap_cut(d=6.0):
    dt = sio.loadmat('overlap_dt_distri_{}.mat'.format(d))['dt'].squeeze()
    cut_dt = sio.loadmat('overlap_cut_dt_{}.mat'.format(d))['cut_dt'].squeeze()
    cut_i = sio.loadmat('overlap_cut_dt_{}.mat'.format(d))['cut_i'].squeeze()
    dt=np.concatenate((dt,cut_dt))
    # t_series=sio.loadmat('t_series_{}.mat'.format(d))['t_series'].squeeze()
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlabel(r'Time $t(\tau)$', fontsize=12)
    ax1.set_ylabel(r'$p(t)$', fontsize=12)
    for point_num in [5000, 10000, 50000, 100000]:
        h_x_eq, h_y_eq = hist_eq_num_per_bin(dt, point_num, True, './hist/cut_hist_eq_num{}_{}.mat'.format(point_num, d))
        ax1.loglog(h_x_eq, h_y_eq, '.', label='bin size={}'.format(point_num))
    xs2 = np.logspace(3, 4, 10)
    ys2 = np.logspace(-5, -7, 10)
    xs1 = np.logspace(4, 5, 10)
    ys1 = np.logspace(2, 3, 10)
    ax1.loglog(xs2, ys2)
    # ax1.loglog(xs1, ys1)
    ax1.text(10 ** (3.5), 10 ** (-5.5), r'~$t^{-2}$')
    # ax1.text(10 ** 4.1, 10 ** 2.5, r'~t')
    ax1.legend()
    fig1.savefig('cut_t_distri_eq_{}.png'.format(d), dpi=500, bbox_inches='tight')
    hist, bins = np.histogram(np.log10(dt), 50, density=False)
    bins_len = np.power(10.0, (bins[1::])) - np.power(10.0, (bins[0:(bins.shape[0] - 1)]))
    # bins_len=10^(bins[1::]-bins[0:(bins.shape[0]-1)])
    h_y = hist / bins_len / dt.shape[0]
    h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_xlabel(r'Time $t(\tau)$', fontsize=12)
    ax2.set_ylabel(r'$p(t)$', fontsize=12)
    ax2.semilogy(h_x, h_y, '.')
    fig2.savefig('cut_t_distri_{}.png'.format(d), dpi=500, bbox_inches='tight')
    ax1.loglog(np.power(10.0, h_x), h_y, '.', label='equal bin length')
    ax1.legend()
    fig1.savefig('cut_t_distri_cmp_{}.png'.format(d), dpi=500, bbox_inches='tight')
    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    ax3.set_xlabel(r'Time $t(\tau)$', fontsize=12)
    ax3.set_ylabel(r'$cut$', fontsize=12)
    ax3.plot(cut_i, [1]*cut_i.shape[0], '.')
    print(dt.shape[0])
    fig3.savefig('cut_i_{}.png'.format(d), dpi=500, bbox_inches='tight')

def length_distri_cal_overlap(time_interval=[1000,5000,10000,50000,100000],init_part=0.1):
    data = sio.loadmat('particles.mat')['data']
    data = data[int(init_part * data.shape[0])::]
    d_num=data.shape[0]
    for ti in time_interval:
        dx=data[0:(d_num-ti),0]-data[ti:d_num,0]
        dy=data[0:(d_num-ti),1]-data[ti:d_num,1]
        dz=data[0:(d_num-ti),2]-data[ti:d_num,2]
        dl=np.sqrt(dx*dx+dy*dy+dz*dz)
        sio.savemat('length_distri_overlap_{}.mat'.format(ti), {'dx': dx,'dl': dl}, do_compression=True)
        # sio.savemat('length_distri_overlap_{}.mat'.format(ti), {'dx': dx,'dy': dy,'dz': dz,'dl': dl}, do_compression=True)
    pass
def length_distri_fig_overlap(time_interval=[1000,5000,10000,50000,100000], binsize=[5000]): #,10000,50000,100000
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlabel(r'step length $x(\sigma)$', fontsize=12)
    ax1.set_ylabel(r'$p(x)$', fontsize=12)
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot()
    # ax2.set_xlabel(r'step length $y(\sigma)$', fontsize=12)
    # ax2.set_ylabel(r'$p(y)$', fontsize=12)
    # fig3 = plt.figure()
    # ax3 = fig3.add_subplot()
    # ax3.set_xlabel(r'step length $z(\sigma)$', fontsize=12)
    # ax3.set_ylabel(r'$p(z)$', fontsize=12)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot()
    ax4.set_xlabel(r'step length $l(\sigma)$', fontsize=12)
    ax4.set_ylabel(r'$4 \pi r^2p(l)$', fontsize=12)
    fig5= plt.figure()
    ax5 = fig5.add_subplot()
    ax5.set_xlabel(r'step length $l(\sigma)$', fontsize=12)
    ax5.set_ylabel(r'$p(l)$', fontsize=12)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_xlabel(r'step length $x(\sigma)$', fontsize=12)
    ax2.set_ylabel(r'$p(x)$', fontsize=12)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    ax3.set_xlabel(r'step length $l(\sigma)$', fontsize=12)
    ax3.set_ylabel(r'$4 \pi p(l)$', fontsize=12)
    fig6 = plt.figure()
    ax6 = fig6.add_subplot()
    ax6.set_xlabel(r'step length $l(\sigma)$', fontsize=12)
    ax6.set_ylabel(r'$p(l)$', fontsize=12)
    for ti in time_interval:
        filename = 'length_distri_overlap_{}.mat'.format(ti)
        dx = sio.loadmat(filename)['dx'].squeeze()
        # dy = sio.loadmat(filename)['dy'].squeeze()
        # dz = sio.loadmat(filename)['dz'].squeeze()
        dl = sio.loadmat(filename)['dl'].squeeze()
        for point_num in binsize:
            h_x_eq, h_y_eq = hist_eq_num_per_bin(dx, point_num, True,
                                                 './hist/dx_hist_eq_num{}_{}_ov.mat'.format(point_num, ti))
            ax1.semilogy(h_x_eq, h_y_eq, '.', label='ti={}'.format(ti))

            # h_x_eq, h_y_eq = hist_eq_num_per_bin(dy, point_num, True,
            #                                      './hist/dy_hist_eq_num{}_{}.mat'.format(point_num, ti))
            # ax2.semilogy(h_x_eq, h_y_eq, '.', label='bin size={}'.format(point_num))
            # h_x_eq, h_y_eq = hist_eq_num_per_bin(dz, point_num, True,
            #                                      './hist/dz_hist_eq_num{}_{}.mat'.format(point_num, ti))
            # ax3.semilogy(h_x_eq, h_y_eq, '.', label='bin size={}'.format(point_num))
            h_x_eq, h_y_eq = hist_eq_num_per_bin(dl, point_num, True,
                                                 './hist/dl_hist_eq_num{}_{}_ov.mat'.format(point_num, ti))
            ax4.plot(h_x_eq, h_y_eq, '.', label='ti={}'.format(ti))
            ax5.semilogy(h_x_eq, h_y_eq/(4*3.14*h_x_eq*h_x_eq), '.', label='ti={}'.format(ti)) #


        hist, bins = np.histogram(dx, 1000, density=True,range=(-50,50))
        # bins_len = np.power(10.0, (bins[1::])) - np.power(10.0, (bins[0:(bins.shape[0] - 1)]))
        # h_y = hist / bins_len / dt.shape[0]
        h_y = hist
        h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        ax2.semilogy(h_x, h_y, '.',label='ti={}'.format(ti))
        sio.savemat('./hist/dx_hist_{}_ov.mat'.format(ti), {'h_x': h_x, 'h_y': h_y}, do_compression=True)
        # hist, bins = np.histogram(dy, 50, density=True)
        # # bins_len = np.power(10.0, (bins[1::])) - np.power(10.0, (bins[0:(bins.shape[0] - 1)]))
        # # h_y = hist / bins_len / dt.shape[0]
        # h_y = hist
        # h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot()
        # ax2.set_xlabel(r'step length $y(\sigma)$', fontsize=12)
        # ax2.set_ylabel(r'$p(y)$', fontsize=12)
        # ax2.semilogy(h_x, h_y, '.')
        # fig2.savefig('dy_distri_{}.png'.format(ti), dpi=500, bbox_inches='tight')
        #
        # hist, bins = np.histogram(dz, 50, density=True)
        # # bins_len = np.power(10.0, (bins[1::])) - np.power(10.0, (bins[0:(bins.shape[0] - 1)]))
        # # h_y = hist / bins_len / dt.shape[0]
        # h_y = hist
        # h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot()
        # ax2.set_xlabel(r'step length $z(\sigma)$', fontsize=12)
        # ax2.set_ylabel(r'$p(z)$', fontsize=12)
        # ax2.semilogy(h_x, h_y, '.')
        # fig2.savefig('dz_distri_{}.png'.format(ti), dpi=500, bbox_inches='tight')

        hist, bins = np.histogram(dl, 1000, density=True,range=(0,100))
        # bins_len = np.power(10.0, (bins[1::])) - np.power(10.0, (bins[0:(bins.shape[0] - 1)]))
        # h_y = hist / bins_len / dt.shape[0]
        h_y = hist
        h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        ax3.plot(h_x, h_y, '.',label='ti={}'.format(ti))
        ax6.semilogy(h_x, h_y/(4*3.14*h_x*h_x), '.',label='ti={}'.format(ti))
        sio.savemat('./hist/dl_hist_{}_ov.mat'.format(ti), {'h_x': h_x, 'h_y': h_y}, do_compression=True)

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()

    fig1.savefig('overlap_dx_distri_eq.png', dpi=500, bbox_inches='tight')
    # fig2.savefig('dy_distri_eq_{}.png'.format(ti), dpi=500, bbox_inches='tight')
    # fig3.savefig('dz_distri_eq_{}.png'.format(ti), dpi=500, bbox_inches='tight')
    fig4.savefig('overlap_dl_distri_eq.png', dpi=500, bbox_inches='tight')
    fig5.savefig('overlap_dl_distri_eq_gs.png', dpi=500, bbox_inches='tight')

    fig2.savefig('overlap_dx_distri.png', dpi=500, bbox_inches='tight')
    fig3.savefig('overlap_dl_distri.png', dpi=500, bbox_inches='tight')
    fig6.savefig('overlap_dl_distri_gs.png', dpi=500, bbox_inches='tight')
def length_distri(time_interval=[1000,5000,10000,50000,100000],init_part=0.1, binsize=[5000]):
    data = sio.loadmat('particles.mat')['data']
    data = data[int(init_part * data.shape[0])::]
    d_num = data.shape[0]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlabel(r'step length $x(\sigma)$', fontsize=12)
    ax1.set_ylabel(r'$p(x)$', fontsize=12)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot()
    ax4.set_xlabel(r'step length $l(\sigma)$', fontsize=12)
    ax4.set_ylabel(r'$4 \pi r^2p(l)$', fontsize=12)
    fig5 = plt.figure()
    ax5 = fig5.add_subplot()
    ax5.set_xlabel(r'step length $l(\sigma)$', fontsize=12)
    ax5.set_ylabel(r'$p(l)$', fontsize=12)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_xlabel(r'step length $x(\sigma)$', fontsize=12)
    ax2.set_ylabel(r'$p(x)$', fontsize=12)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    ax3.set_xlabel(r'step length $l(\sigma)$', fontsize=12)
    ax3.set_ylabel(r'$4 \pi p(l)$', fontsize=12)
    fig6 = plt.figure()
    ax6 = fig6.add_subplot()
    ax6.set_xlabel(r'step length $l(\sigma)$', fontsize=12)
    ax6.set_ylabel(r'$p(l)$', fontsize=12)
    for ti in time_interval:
        dx = data[0:(d_num - ti), 0] - data[ti:d_num, 0]
        dy = data[0:(d_num - ti), 1] - data[ti:d_num, 1]
        dz = data[0:(d_num - ti), 2] - data[ti:d_num, 2]
        dl = np.sqrt(dx * dx + dy * dy + dz * dz)
        for point_num in binsize:
            h_x_eq, h_y_eq = hist_eq_num_per_bin(dx, point_num, True,
                                                 './hist/dx_hist_eq_num{}_{}_ov.mat'.format(point_num, ti))
            ax1.semilogy(h_x_eq, h_y_eq, '.', label='ti={}'.format(ti))
            h_x_eq, h_y_eq = hist_eq_num_per_bin(dl, point_num, True,
                                                 './hist/dl_hist_eq_num{}_{}_ov.mat'.format(point_num, ti))
            ax4.plot(h_x_eq, h_y_eq, '.', label='ti={}'.format(ti))
            ax5.semilogy(h_x_eq, h_y_eq / (4 * 3.14 * h_x_eq * h_x_eq), '.', label='ti={}'.format(ti))  #

        hist, bins = np.histogram(dx, 1000, density=True, range=(-50, 50))
        h_y = hist
        h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        ax2.semilogy(h_x, h_y, '.', label='ti={}'.format(ti))
        sio.savemat('./hist/dx_hist_{}_ov.mat'.format(ti), {'h_x': h_x, 'h_y': h_y}, do_compression=True)


        hist, bins = np.histogram(dl, 1000, density=True, range=(0, 100))
        h_y = hist
        h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        ax3.plot(h_x, h_y, '.', label='ti={}'.format(ti))
        ax6.semilogy(h_x, h_y / (4 * 3.14 * h_x * h_x), '.', label='ti={}'.format(ti))
        sio.savemat('./hist/dl_hist_{}_ov.mat'.format(ti), {'h_x': h_x, 'h_y': h_y}, do_compression=True)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()

    fig1.savefig('overlap_dx_distri_eq.png', dpi=500, bbox_inches='tight')
    # fig2.savefig('dy_distri_eq_{}.png'.format(ti), dpi=500, bbox_inches='tight')
    # fig3.savefig('dz_distri_eq_{}.png'.format(ti), dpi=500, bbox_inches='tight')
    fig4.savefig('overlap_dl_distri_eq.png', dpi=500, bbox_inches='tight')
    fig5.savefig('overlap_dl_distri_eq_gs.png', dpi=500, bbox_inches='tight')

    fig2.savefig('overlap_dx_distri.png', dpi=500, bbox_inches='tight')
    fig3.savefig('overlap_dl_distri.png', dpi=500, bbox_inches='tight')
    fig6.savefig('overlap_dl_distri_gs.png', dpi=500, bbox_inches='tight')

def length_distri_data_delete_overlap(time_interval=[1000,5000,10000,50000,100000]):
    for ti in time_interval:
        os.remove('length_distri_overlap_{}.mat'.format(ti))

def length_distri_cal(time_interval=[1000,5000,10000,50000,100000],init_part=0.1):
    data = sio.loadmat('particles.mat')['data']
    data = data[int(init_part * data.shape[0])::]
    d_num=data.shape[0]
    for ti in time_interval:
        r0=data[0]
        i=ti
        dx=[]
        dy=[]
        dz=[]
        dl=[]
        while i<d_num:
            dx+=[data[i][0]-r0[0]]
            dy+=[data[i][1]-r0[1]]
            dz+=[data[i][2]-r0[2]]
            dl+=[np.linalg.norm(data[i]-r0)]
            r0=data[i]
            i+=ti
        # uni_dx=np.unique(dx)
        sio.savemat('length_distri_{}.mat'.format(ti), {'dx': dx,'dy': dy,'dz': dz,'dl': dl}, do_compression=True)
    pass
def length_distri_fig(time_interval=[1000,5000,10000,50000,100000], binsize=[10]): #,10000,50000,100000
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlabel(r'step length $x(\sigma)$', fontsize=12)
    ax1.set_ylabel(r'$p(x)$', fontsize=12)
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot()
    # ax2.set_xlabel(r'step length $y(\sigma)$', fontsize=12)
    # ax2.set_ylabel(r'$p(y)$', fontsize=12)
    # fig3 = plt.figure()
    # ax3 = fig3.add_subplot()
    # ax3.set_xlabel(r'step length $z(\sigma)$', fontsize=12)
    # ax3.set_ylabel(r'$p(z)$', fontsize=12)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot()
    ax4.set_xlabel(r'step length $l(\sigma)$', fontsize=12)
    ax4.set_ylabel(r'$4 \pi r^2p(l)$', fontsize=12)
    fig5= plt.figure()
    ax5 = fig5.add_subplot()
    ax5.set_xlabel(r'step length $l(\sigma)$', fontsize=12)
    ax5.set_ylabel(r'$p(l)$', fontsize=12)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_xlabel(r'step length $x(\sigma)$', fontsize=12)
    ax2.set_ylabel(r'$p(x)$', fontsize=12)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    ax3.set_xlabel(r'step length $l(\sigma)$', fontsize=12)
    ax3.set_ylabel(r'$4 \pi r^2p(l)$', fontsize=12)
    fig6 = plt.figure()
    ax6 = fig6.add_subplot()
    ax6.set_xlabel(r'step length $l(\sigma)$', fontsize=12)
    ax6.set_ylabel(r'$p(l)$', fontsize=12)
    for ti in time_interval:
        filename = 'length_distri_{}.mat'.format(ti)
        dx = sio.loadmat(filename)['dx'].squeeze()
        # dy = sio.loadmat(filename)['dy'].squeeze()
        # dz = sio.loadmat(filename)['dz'].squeeze()
        dl = sio.loadmat(filename)['dl'].squeeze()
        for point_num in binsize:
            h_x_eq, h_y_eq = hist_eq_num_per_bin(dx, point_num, True,
                                                 './hist/dx_hist_eq_num{}_{}.mat'.format(point_num, ti))
            ax1.semilogy(h_x_eq, h_y_eq, '.', label='ti={}'.format(ti))

            # h_x_eq, h_y_eq = hist_eq_num_per_bin(dy, point_num, True,
            #                                      './hist/dy_hist_eq_num{}_{}.mat'.format(point_num, ti))
            # ax2.semilogy(h_x_eq, h_y_eq, '.', label='bin size={}'.format(point_num))
            # h_x_eq, h_y_eq = hist_eq_num_per_bin(dz, point_num, True,
            #                                      './hist/dz_hist_eq_num{}_{}.mat'.format(point_num, ti))
            # ax3.semilogy(h_x_eq, h_y_eq, '.', label='bin size={}'.format(point_num))
            h_x_eq, h_y_eq = hist_eq_num_per_bin(dl, point_num, True,
                                                 './hist/dl_hist_eq_num{}_{}.mat'.format(point_num, ti))
            ax4.plot(h_x_eq, h_y_eq, '.', label='ti={}'.format(ti))
            ax5.semilogy(h_x_eq, h_y_eq/(4*3.14*h_x_eq*h_x_eq), '.', label='ti={}'.format(ti)) #


        hist, bins = np.histogram(dx, 1000, density=True,range=(-50,50))
        # bins_len = np.power(10.0, (bins[1::])) - np.power(10.0, (bins[0:(bins.shape[0] - 1)]))
        # h_y = hist / bins_len / dt.shape[0]
        h_y = hist
        # print(np.sum(h_y))
        h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        ax2.semilogy(h_x, h_y, '.',label='ti={}'.format(ti))
        sio.savemat('./hist/dx_hist_{}.mat'.format(ti), {'h_x': h_x, 'h_y': h_y}, do_compression=True)
        # hist, bins = np.histogram(dy, 50, density=True)
        # # bins_len = np.power(10.0, (bins[1::])) - np.power(10.0, (bins[0:(bins.shape[0] - 1)]))
        # # h_y = hist / bins_len / dt.shape[0]
        # h_y = hist
        # h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot()
        # ax2.set_xlabel(r'step length $y(\sigma)$', fontsize=12)
        # ax2.set_ylabel(r'$p(y)$', fontsize=12)
        # ax2.semilogy(h_x, h_y, '.')
        # fig2.savefig('dy_distri_{}.png'.format(ti), dpi=500, bbox_inches='tight')
        #
        # hist, bins = np.histogram(dz, 50, density=True)
        # # bins_len = np.power(10.0, (bins[1::])) - np.power(10.0, (bins[0:(bins.shape[0] - 1)]))
        # # h_y = hist / bins_len / dt.shape[0]
        # h_y = hist
        # h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot()
        # ax2.set_xlabel(r'step length $z(\sigma)$', fontsize=12)
        # ax2.set_ylabel(r'$p(z)$', fontsize=12)
        # ax2.semilogy(h_x, h_y, '.')
        # fig2.savefig('dz_distri_{}.png'.format(ti), dpi=500, bbox_inches='tight')

        hist, bins = np.histogram(dl, 1000, density=True,range=(0,100))
        # bins_len = np.power(10.0, (bins[1::])) - np.power(10.0, (bins[0:(bins.shape[0] - 1)]))
        # h_y = hist / bins_len / dt.shape[0]
        h_y = hist
        h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        ax3.plot(h_x, h_y, '.',label='ti={}'.format(ti))
        ax6.semilogy(h_x, h_y/(4*3.14*h_x*h_x), '.',label='ti={}'.format(ti))
        sio.savemat('./hist/dl_hist_{}.mat'.format(ti), {'h_x': h_x, 'h_y': h_y}, do_compression=True)

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()

    fig1.savefig('dx_distri_eq.png', dpi=500, bbox_inches='tight')
    # fig2.savefig('dy_distri_eq_{}.png'.format(ti), dpi=500, bbox_inches='tight')
    # fig3.savefig('dz_distri_eq_{}.png'.format(ti), dpi=500, bbox_inches='tight')
    fig4.savefig('dl_distri_eq.png', dpi=500, bbox_inches='tight')
    fig5.savefig('dl_distri_eq_gs.png', dpi=500, bbox_inches='tight')

    fig2.savefig('dx_distri.png', dpi=500, bbox_inches='tight')
    fig3.savefig('dl_distri.png', dpi=500, bbox_inches='tight')
    fig6.savefig('dl_distri_gs.png', dpi=500, bbox_inches='tight')
def length_distri_data_delete(time_interval=[1000,5000,10000,50000,100000]):
    for ti in time_interval:
        os.remove('length_distri_{}.mat'.format(ti))

def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0
def wavelet_noise_analyse(init_part=0.1,level=10,down_sample=1, lamda_k=1.0):
    data = sio.loadmat('particles.mat')['data']
    data = (data[int(init_part * data.shape[0])::])[::down_sample]
    ts=np.arange(data.shape[0])*down_sample
    d_num=data.shape[0]
    dim=['x','y','z']
    d=0
    while d<len(dim):
        r=data[:,d].tolist()
        w = pywt.Wavelet('haar')
        CA_CD = pywt.wavedec(r, w, level=level)  # [ca_level,cd_level...cd1]
        median_cd1 = np.median(np.abs(CA_CD[level]))
        sigma = (1.0 / 0.6745) * median_cd1 * 3
        lamda = sigma * math.sqrt(2.0 * math.log(float(d_num), math.e))*lamda_k
        usecoeffs = []
        usecoeffs.append(CA_CD[0])
        a = 0
        i = 1
        while i <= level:
            cd_i = CA_CD[i]
            length_i = len(cd_i)
            cd_i[cd_i <= lamda] = 0.0
            # for k in range(length_i):
            #     if (abs(cd_i[k]) >= lamda):
            #         cd_i[k] = sgn(cd_i[k]) * (abs(cd_i[k]) - a * lamda)
            #     else:
            #         cd_i[k] = 0.0
            usecoeffs.append(cd_i)
            i += 1
        r_new = pywt.waverec(usecoeffs, w)[0:d_num]
        fig1 = plt.figure()
        ax1 = fig1.add_subplot()
        ax1.set_xlabel(r'Time Step', fontsize=12)
        ax1.set_ylabel(r'{}'.format(dim[d]), fontsize=12)
        ax1.plot(ts,r,linewidth=1)
        ax1.plot(ts,r_new,linewidth=1)
        fig1.savefig('trace_{}.png'.format(dim[d]), dpi=500, bbox_inches='tight')
        # sio.savemat('r_{}.mat'.format(dim[d]), {'r_new': r_new, 'ts': ts})
        Distri_cal(r_new, dim[d])
        d += 1


    pass
def Distri_cal(r_new,dim):
    r_num=len(r_new)
    i=1
    i0=0
    time=[]
    length=[]
    while i < r_num:
        delta_r=r_new[i]-r_new[i0]
        if np.abs(delta_r) > 1e-5:
            time+=[i-i0]
            length+=[delta_r]
            i0=i
        i+=1
    hist_time(time,'time_{}'.format(dim))
    hist_fig(length,'length_{}'.format(dim))
    pass
def hist_fig(dd,name,x_label=r'Time $t(\tau)$',y_label=r'$p(t)$',bin_size=[5, 10, 20, 40, 80, 200, 500],bin_num=200,log=True):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel(y_label, fontsize=12)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel(y_label, fontsize=12)
    for point_num in bin_size:
        h_x_eq, h_y_eq = hist_eq_num_per_bin(dd, point_num, True, './hist/hist_{}.mat'.format(name))
        if log:
            ax1.loglog(h_x_eq, h_y_eq, '.', label='bin size={}'.format(point_num))
        else:
            ax1.semilogy(h_x_eq, h_y_eq, '.', label='bin size={}'.format(point_num))
    # xs2 = np.logspace(3, 4, 10)
    # ys2 = np.logspace(-5, -7, 10)
    # xs1 = np.logspace(4, 5, 10)
    # ys1 = np.logspace(2, 3, 10)
    # ax1.loglog(xs2, ys2)
    # # ax1.loglog(xs1, ys1)
    # ax1.text(10 ** (3.5), 10 ** (-5.5), r'~$t^{-2}$')
    # ax1.text(10 ** 4.1, 10 ** 2.5, r'~t')
    ax1.legend()
    fig1.savefig('{}_eq.png'.format(name), dpi=500, bbox_inches='tight')
    if log:
        hist, bins = np.histogram(np.log10(dd), bin_num, density=False)
        bins_len = np.power(10.0, (bins[1::])) - np.power(10.0, (bins[0:(bins.shape[0] - 1)]))
        # bins_len=10^(bins[1::]-bins[0:(bins.shape[0]-1)])
        h_y = hist / bins_len / len(dd)
        h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        ax2.semilogy(h_x, h_y, '.')
        ax1.loglog(np.power(10.0, h_x), h_y, '.', label='equal bin length')
        ax1.legend()
        fig1.savefig('{}_cmp.png'.format(name), dpi=500, bbox_inches='tight')
    else:
        hist, bins = np.histogram(dd, 200, density=True)
        h_y = hist
        h_x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        ax2.semilogy(h_x, h_y, '.')
        ax1.semilogy(h_x, h_y, '.', label='equal bin length')
        ax1.legend()
        fig1.savefig('{}_cmp.png'.format(name), dpi=500, bbox_inches='tight')
    fig2.savefig('{}.png'.format(name), dpi=500, bbox_inches='tight')
def hist_time_forward(tt, name,x_label=r'Time $t(\tau)$',y_label=r'$p(t)$'):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel(y_label, fontsize=12)
    delta_t=[]
    dt_num=[]
    t_num=len(tt)
    tt=np.sort(tt)
    # hist_x+=tt[0]
    t0=tt[0]
    sum=1.0
    i=1
    while i<t_num:
        if tt[i]==t0:
            i+=1
            sum+=1.0
        else:
            delta_t+=[t0]
            dt_num+=[sum]  #/float(t_num)
            t0=tt[i]
            i+=1
            sum=1.0
    min_binsize=10
    h_x=[]
    h_y=[]
    i=0
    k=0
    j=1
    y_sum=0
    y_remain=np.sum(dt_num)
    while k<len(delta_t):
        y_sum+=dt_num[k]
        y_remain-=dt_num[k]
        if y_remain < min_binsize:
            j=len(delta_t)-1
            y_sum+=y_remain
            y_sum-=dt_num[j]
            h_x += [(delta_t[i] + delta_t[j]) / 2.0]
            h_y += [y_sum / float(len(delta_t)) / (delta_t[j] - delta_t[i])]
            break
        if y_sum < min_binsize:
            k+=1
            j+=1
            pass
        else:
            h_x+=[(delta_t[i]+delta_t[j])/2.0]
            h_y+=[y_sum/float(len(delta_t))/(delta_t[j]-delta_t[i])]
            i=j
            k=i
            j+=1
            y_sum=0
    ax1.loglog(h_x, h_y, '.')
    fig1.savefig('{}.png'.format(name), dpi=500, bbox_inches='tight')
def hist_time(tt, name,x_label=r'Time $t(\tau)$',y_label=r'$p(t)$'):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel(y_label, fontsize=12)
    delta_t=[]
    dt_num=[]
    t_num=len(tt)
    tt=np.sort(tt)
    # hist_x+=tt[0]
    t0=tt[0]
    sum=1.0
    i=1
    while i<t_num:
        if tt[i]==t0:
            i+=1
            sum+=1.0
        else:
            delta_t+=[t0]
            dt_num+=[sum]  #/float(t_num)
            t0=tt[i]
            i+=1
            sum=1.0
    min_binsize=10
    delta_t=[0]+delta_t
    dt_num=[0]+dt_num
    h_x=[]
    h_y=[]
    k=1
    j=0
    y_sum=0
    y_remain=np.sum(dt_num)
    while k<len(delta_t):
        y_sum+=dt_num[k]
        y_remain-=dt_num[k]
        if y_remain < min_binsize:
            k=len(delta_t)-1
            y_sum+=y_remain
            h_x += [(delta_t[j] + delta_t[k]) / 2.0]
            h_y += [y_sum / float(len(delta_t)) / (delta_t[k] - delta_t[j])]
            break
        if y_sum < min_binsize:
            k+=1
            pass
        else:
            h_x+=[(delta_t[j]+delta_t[k])/2.0]
            h_y+=[y_sum/float(len(delta_t))/(delta_t[k]-delta_t[j])]
            j=k
            k+=1
            y_sum=0
    ax1.loglog(h_x, h_y, '.')
    fig1.savefig('{}.png'.format(name), dpi=500, bbox_inches='tight')

def pdf_one_dimention(dimention=0,init_part=0.1):
    bin_size=0.1
    bin_range=(-100,100)
    bin_num=int((bin_range[1]-bin_range[0])/bin_size)

    data = sio.loadmat('particles.mat')['data']
    data = data[int(init_part * data.shape[0])::]
    d_num = data.shape[0]
    time_interval=np.array([200,2000,5000,10000,20000,80000,200000,400000,800000,2000000]) #
    print(data.shape)
    i=0
    data = torch.from_numpy(data).type(torch.Tensor).cuda()
    pdf = torch.zeros((time_interval.shape[0],bin_num)).cuda()
    pdf_nor = torch.zeros((time_interval.shape[0],bin_num)).cuda()
    for ti in time_interval:
        dx = data[0:(d_num - ti), dimention] - data[ti:d_num, dimention]
        dx_nor=dx/(ti**0.5)*500
        pdf[i,:] = torch.histc(dx, bin_num, min=bin_range[0],max=bin_range[1])/dx.shape[0]/bin_size
        pdf_nor[i,:] = torch.histc(dx_nor, bin_num, min=bin_range[0],max=bin_range[1])/dx.shape[0]/bin_size
        i += 1
        if i%100==0:
            print(i)
    pdf=pdf.cpu().numpy()
    pdf_nor=pdf_nor.cpu().numpy()
    bins=np.arange(bin_range[0],bin_range[1]+bin_size,bin_size)
    x=0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
    sio.savemat('./pdf_x.mat', {'pdf': pdf, 't': time_interval,'x': x, 'pdf_nor':pdf_nor}, do_compression=True)
    pass

def pdf_r(init_part=0.1):
    bin_size=0.1
    bin_range=(0,100)
    bin_num=int((bin_range[1]-bin_range[0])/bin_size)

    data = sio.loadmat('particles.mat')['data']
    data = data[int(init_part * data.shape[0])::]
    d_num = data.shape[0]
    time_interval=np.array([200,2000,5000,10000,20000,80000,200000,400000,1000000,2000000]) #

    i=0
    data = torch.from_numpy(data).type(torch.Tensor).cuda()
    pdf = torch.zeros((time_interval.shape[0],bin_num)).cuda()
    pdf_nor = torch.zeros((time_interval.shape[0],bin_num)).cuda()
    for ti in time_interval:
        dxyz2=(data[0:(d_num - ti)] - data[ti:d_num])**2
        dr=torch.sqrt(torch.sum(dxyz2, dim=1)).squeeze()
        dr_nor=dr/(ti**0.5)*500
        pdf[i,:] = torch.histc(dr, bin_num, min=bin_range[0],max=bin_range[1])/dr.shape[0]/bin_size
        pdf_nor[i,:] = torch.histc(dr_nor, bin_num, min=bin_range[0],max=bin_range[1])/dr.shape[0]/bin_size
        i += 1
        if i%100==0:
            print(i)
    pdf=pdf.cpu().numpy()
    pdf_nor=pdf_nor.cpu().numpy()
    bins=np.arange(bin_range[0],bin_range[1]+bin_size,bin_size)
    r=0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
    sio.savemat('./pdf_r.mat', {'pdf': pdf, 't': time_interval,'r': r,'pdf_nor':pdf_nor}, do_compression=True)
    pass
def pdf_fig():
    pdf = sio.loadmat('pdf_x.mat')['pdf']
    t=sio.loadmat('pdf_x.mat')['t']
    x=sio.loadmat('pdf_x.mat')['x']
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('p')
    # plt.zscale('log')
    X,T=np.meshgrid(x,t)
    PDF=np.log10(pdf)
    ax.contour3D(X,T,PDF,50,cmap='jet')
    # ax.plot_surface(X, T, PDF, rstride=1, cstride=1,
    #              cmap='viridis', edgecolor='none')
    # plt.show()
    fig.savefig('pdf.png', dpi=500, bbox_inches='tight')
def pdf_fig_part():
    pdf = sio.loadmat('pdf_x.mat')['pdf']
    t = sio.loadmat('pdf_x.mat')['t'].squeeze()
    x = sio.loadmat('pdf_x.mat')['x'].squeeze()
    fig = plt.figure()
    ax=fig.add_subplot()
    ax.set_xlabel(r'$x(\sigma)$', fontsize=12)
    ax.set_ylabel(r'$p(x)$', fontsize=12)
    # time_interval=[1,10,100,1000,5000,9000]
    ii=0
    for ti in t:
        ax.semilogy(x,pdf[ii,:],'.',label='ti={}'.format(ti))
        ii += 1
    ax.legend()
    fig.savefig('pdf_part.png', dpi=500, bbox_inches='tight')
def pdf_fig_r_part():
    pdf = sio.loadmat('pdf_r.mat')['pdf']
    t = sio.loadmat('pdf_r.mat')['t'].squeeze()
    r = sio.loadmat('pdf_r.mat')['r'].squeeze()
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel(r'$r(\sigma)$', fontsize=12)
    ax.set_ylabel(r'$4\pi r^2p(r)$', fontsize=12)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_xlabel(r'$r(\sigma)$', fontsize=12)
    ax2.set_ylabel(r'$p(r)$', fontsize=12)
    time_interval = [100, 1000, 5000, 9000]
    ii=0
    for ti in t:
        ax.plot(r, pdf[ii, :], '.', label='ti={}'.format(ti))
        ax2.semilogy(r, pdf[ii, :]/(4*np.pi*r*r), '.', label='ti={}'.format(ti))
        ii+=1
    ax.legend()
    ax2.legend()
    fig.savefig('pdf_r_part.png', dpi=500, bbox_inches='tight')
    fig2.savefig('pdf_r_part2.png', dpi=500, bbox_inches='tight')
def pdf_one_dimention_numpy(dimention=0,init_part=0.1):
    bin_size=0.1
    bin_range=(-100,100)
    bin_num=int((bin_range[1]-bin_range[0])/bin_size)
    data = sio.loadmat('particles.mat')['data']
    data = data[int(init_part * data.shape[0])::]
    d_num = data.shape[0]
    time_interval=np.arange(0,d_num//10,1000).astype(int)
    pdf=[]
    i=0
    # data = torch.from_numpy(data).type(torch.Tensor).cuda()
    for ti in time_interval:
        i+=1
        dx = data[0:(d_num - ti), dimention] - data[ti:d_num, 0]
        hist, bins = np.histogram(dx, bin_num, density=True, range=bin_range)
        pdf+=[hist]
        # x = 0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
        if i%100==0:
            print(i)

    x=0.5 * (bins[1::] + bins[0:(bins.shape[0] - 1)])
    sio.savemat('./pdf_x.mat', {'pdf': pdf, 't': time_interval,'x': x}, do_compression=True)
    pass


if __name__ == '__main__':
    # time_distri_cal_overlap()
    # time_distri_fig_overlap()
    # wavelet_noise_analyse(0.1,lamda_k=1.0)
    # time_distri_cal()
    # time_distri_fig()

    pdf_one_dimention()
    pdf_fig_part()
    # pdf_fig()
    pdf_r()
    pdf_fig_r_part()
    #
    # length_distri(time_interval=[1000,5000,10000,50000,100000,500000,1000000],init_part=0.01)
    # length_distri_cal_overlap(time_interval=[1000,5000,10000,50000,100000,500000,1000000])
    # length_distri_fig_overlap(time_interval=[1000,5000,10000,50000,100000,500000,1000000])
    # length_distri_data_delete_overlap(time_interval=[1000,5000,10000,50000,100000,500000,1000000])
    pass
