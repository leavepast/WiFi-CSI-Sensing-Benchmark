from decoders import interleaved as decoder
import glob
import loguru as logger
import matplotlib.pyplot as plt
import numpy as np
import csiread
import numpy as np
import os
from tqdm import tqdm

save_path=f"/data/wifi/hexin/orgin_cutted_2"
if not os.path.exists(save_path):
    os.makedirs(save_path)

list=glob.glob('/data/wifi/hexin/orgin_2/*/*.txt')
period=101
#数据截取
for file in tqdm(list):
    csi=np.loadtxt(file)
    task = file.split('/')[-1].split('.')[0]
    stu = file.split('/')[5].split('_')[0]
    #创建人员文件夹
    stu_path = os.path.join(save_path, stu)
    if not os.path.exists(stu_path):
        os.makedirs(stu_path)
    #创建任务文件夹
    task_path = os.path.join(stu_path, task)
    if not os.path.exists(task_path):
        os.makedirs(task_path)
    #保存文件
    for i in range(30):
        repeat=csi[i*period:(i+1)*period,:]
        np.savetxt(os.path.join(task_path,str(task)+'_'+str(i)+'.txt'),np.abs(repeat))
