from CSIKit.tools.batch_graph import BatchGraph
import matlab.engine
import  numpy as np
import csiread
import numpy as np
import os
import glob
from tqdm import tqdm
save_path=f"/data/wifi/hexin/orgin_2"
if not os.path.exists(save_path):
    os.makedirs(save_path)


eng = matlab.engine.start_matlab()
workDir = eng.genpath('/home/xjw/csi_process/')
eng.addpath(workDir, nargout=0)

if not os.path.exists(save_path):
    os.makedirs(save_path)

list=glob.glob('/data/wifi/hexin/data2/data/*/*.pcap')
print("#total file is ",len(list))
count=0
for file in tqdm(list):
    stu = file.split('/')[6].split('_')[0]
    task = file.split('/')[-1].split('.')[0]
    try:
        task=int(task)
        task='t'+str(task)
    except:
        pass
    # 创建人员文件夹
    stu_path = os.path.join(save_path, stu)
    if not os.path.exists(stu_path):
        os.makedirs(stu_path)
    # 创建任务文件夹
    # task_path = os.path.join(stu_path, task)
    # if not os.path.exists(task_path):
    #     os.makedirs(task_path)

    try:
        csi_amp =np.array(eng.readcsi1(file, 0, 0))
        count=count+1
        np.savetxt(os.path.join(stu_path,str(task)+'.txt'),csi_amp)
    except:
        print('#error-->filename is ',file)