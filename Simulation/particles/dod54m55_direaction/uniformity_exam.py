import numpy as np
from Galamostxmlreader import GalamostXmlreader as GxmlReader
from collections import Counter
import matplotlib.pyplot as plt
from gala_model import polygon
file_name='Particle.xml'
ord_num_file='npnumber.txt'
radius=float(np.loadtxt('radius.txt').astype('float'))
num_of_sample=100000
Particle = GxmlReader(file_name)
pos = Particle.positiondata/radius

# p=polygon()
# pos=p.GetApex('hex')

ord_num = np.array(np.loadtxt(ord_num_file).astype('int').tolist())
particle_pos =pos[1:ord_num[0],:]

a = np.random.random((num_of_sample, 1)) * 2 * np.pi  # 根据随机求面均匀分布，先生成一个初始状态
b = np.arcsin(np.random.random((num_of_sample, 1)) * 2 - 1)
sample = np.concatenate((np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)), axis=-1)

dis = np.linalg.norm(particle_pos[:,None,:]-sample[None,:,:], axis=-1)

min_dis = np.argmin(dis, axis=0)
# group = np.array(Counter(min_dis).most_common())
bins=np.arange(0, particle_pos.shape[0]+1, 1)
fig = plt.figure()
ax = fig.add_subplot()
ax.hist(min_dis+0.1, bins)
fig.savefig('uniform_exam.png',dpi=500,bbox_inches = 'tight')
pass