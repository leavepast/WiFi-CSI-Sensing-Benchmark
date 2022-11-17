from decoders import interleaved as decoder
import glob
import loguru as logger
import matplotlib.pyplot as plt
import numpy as np


#file = "/data/wifi/hexin/data2/data/s1_陈锡敏/t1.pcap"
file = "/data/wifi/hexin/g1/s1/alltxt/t1.txt"
csidata=np.loadtxt(file)
plt.pcolormesh(np.abs(csidata),cmap='RdBu_r')
plt.colorbar()
plt.show()