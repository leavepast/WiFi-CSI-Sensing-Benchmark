import glob
import loguru as logger
import matplotlib.pyplot as plt
import numpy as np
import csiread
import numpy as np


#读取数据
files = glob.glob(f'/data/wifi/hexin/orgin_cutted*/s1/*/t1_0.txt')
fig=plt.figure()
length=len(files)
loc=str(length)+'1'
for index,file in enumerate(files):
    ax2=fig.add_subplot(int(loc+str(index+1)))
    csi=np.loadtxt(file)
    csi
    ax2.pcolormesh(csi,cmap='RdBu_r')
plt.show()