from decoders import interleaved as decoder
import glob
import loguru as logger
import matplotlib.pyplot as plt
import numpy as np
import csiread
import numpy as np

#file = "/data/wifi/hexin/data2/data/s1_陈锡敏/t1.pcap"
# file = "/data/wifi/hexin/data_zu1/s1_高勤标/pcap/t1.pcap"
# list1=glob.glob('/data/wifi/hexin/data_zu1/*/pcap/*.pcap')
# list2=glob.glob('/data/wifi/hexin/data_zu1/*/*.pcap')
# list=list1+list2
# list_ok=[]
# for file in list:
#     try:
#         csidata = csiread.Nexmon(file, chip='4358', bw=80)
#         csidata.read()
#         list_ok.append(file)
#         # ori_data = csidata.csi
#         # samples = decoder.read_pcap(file)
#         # print(ori_data.shape)
#     except:
#         pass
csi=np.loadtxt(f'/data/wifi/hexin/orgin/s6/t10.txt')
plt.pcolormesh(csi,cmap='RdBu_r')
plt.colorbar()
plt.show()
# #切分重复动作
# period=101
# for file in list_ok:
#     csidata = csiread.Nexmon(file, chip='4358', bw=80)
#     csidata.read()
#     csi=csidata.csi
#     for i in range(30):
#         temp=csi[i*period:(i+1)*period]

# csidata = csiread.Nexmon(f't1.pcap', chip='4358', bw=80)
# csidata.read()
# data=csidata.csi
# pass