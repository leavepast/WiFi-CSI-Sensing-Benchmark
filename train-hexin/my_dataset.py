import sys

import numpy as np
import torch
from torch.utils.data import Dataset
import csiread
import  analyse.analyse_wavelet as ana

class HeXin_Dataset(Dataset):
    def __init__(self,data_list):
        self.data_list = data_list
        self.category = {i:i for i in range(0,6)}
        self.current=-1
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        self.current = idx
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_file = self.data_list[idx]
        csidata = csiread.Nexmon(sample_file, chip='4358', bw=80)
        csidata.read()
        csi =csidata.csi
        csi_amp = abs(csi)
        x=csi_amp[0:2500,:]
        x = torch.FloatTensor(x)
        y = self.category[int(sample_file.split('/')[-1].split('-')[1]) - 1]
        return x, y

class HeXin_Txt_Dataset(Dataset):
    def __init__(self,data_list):
        self.data_list = data_list
        self.category = {'t'+str(i):i-1 for i in range(1,21)}
        self.current=-1
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        self.current = idx
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_file = self.data_list[idx]
        csi_amp = np.loadtxt(sample_file, encoding='utf-8')
        #subs_adj = ana.reduce_noise_batch(csi_amp)
        # subs_low=ana.reduce_dim(subs_adj,4,62)
        rows,cols=csi_amp.shape
        if rows < 101:  # 判断文件是否是101行62列，如果小于101行，则生成101-rows行0数据进行填补
            add = np.zeros((101 - rows,cols ))  # 生成的0数组存放到file_add
            csi_amp = np.concatenate((csi_amp, add))  # 沿轴 0 连接两个数组
        x=csi_amp
        x = torch.FloatTensor(x)
        try:
            y = self.category[sample_file.split('/')[-1].split('_')[0]]
        except:
            y = self.category['t'+sample_file.split('/')[-1].split('_')[0]]
        return x.unsqueeze(dim=0), y