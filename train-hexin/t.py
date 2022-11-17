import torch
# 返回gpu数量；
a=torch.cuda.device_count()
for i in range(a):
    # 返回gpu名字，设备索引默认从0开始；
    cuda=torch.cuda.get_device_name(i)
pass