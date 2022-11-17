import os
import math
import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import glob
# from model import HeXin3_LSTM
from model.model_v2 import MobileNetV2
from my_dataset import HeXin_Txt_Dataset
from tqdm import tqdm
import csiread

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #读取数据
    files = glob.glob(f'/data/wifi/hexin/orgin_cutted_1/*/*/*.txt')
    # files=files[:20]
    val_loader = torch.utils.data.DataLoader(dataset=HeXin_Txt_Dataset(files),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0)
    model = MobileNetV2(num_classes=args.num_classes).to(device)
    if args.weights != "":
        weights_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weights_dict,strict=False)

    val_acc  = evaluate(model=model,
                   data_loader=val_loader,
                   device=device,
                    )
    print(val_acc)



@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

    return  accu_num.item() / sample_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                         default="/data/wifi/hexin")
                        #default=r"/home/data1/wifi")
    parser.add_argument('--model-name', default='RegNetY_400MF', help='create model name')
    parser.add_argument('--weights', type=str, default='./weights/RegNetY_400MF/RegNetY_400MF-99.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
