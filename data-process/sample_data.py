import  random
import  glob
import torch
import os
import  shutil
sample_num=10000
random.seed(0)
root_dir=r'E:\1-widar\Widar3.0\CSI\20181109/'
data_list = glob.glob(root_dir + '/*/*.dat')

train_files_path = []
val_files_path = []
files = random.sample(data_list,10000)
train_size = int(0.7 * len(files))
test_size  = int(0.2 * len(files))
val_size   = int(0.1 * len(files))
train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(files, [train_size, test_size, val_size])

# for path in data_list:
#     if path in files:  # 如果该路径在采样的验证集样本中则存入验证集
#         val_files_path.append(path)
#     else:  # 否则存入训练集
#         train_files_path.append(path)
#将训练集标记存入txt文件中
train_file = []

train_target_dir=r'E:\1-widar\Widar3.0\CSI\20181109_ahnu/train'
if not os.path.exists(train_target_dir):
    os.makedirs(train_target_dir)
for line in train_dataset:
    shutil.copy(line, train_target_dir)

test_target_dir=r'E:\1-widar\Widar3.0\CSI\20181109_ahnu/test'
if not os.path.exists(test_target_dir):
    os.makedirs(test_target_dir)
for line in test_dataset:
    shutil.copy(line, test_target_dir)

val_target_dir=r'E:\1-widar\Widar3.0\CSI\20181109_ahnu/val'
if not os.path.exists(val_target_dir):
    os.makedirs(val_target_dir)
for line in val_dataset:
    shutil.copy(line, val_target_dir)


pass