import numpy as np
import torch
import torch.nn as nn
import argparse
from util import load_data_n_model
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from tqdm import tqdm
import sys

def train(model, tensor_loader, num_epochs, learning_rate, criterion, device,tb_writer,tags,logger):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    for epoch in range(num_epochs):
        model.train()
        tensor_loader = tqdm(tensor_loader, file=sys.stdout)

        accu_loss = torch.zeros(1).to(device)  # 累计损失
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        sample_num = 0
        for step,data in enumerate(tensor_loader):
            inputs,labels = data
            sample_num += inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            ##累计损失
            accu_loss += loss.detach()
            predict_y = torch.argmax(outputs,dim=1).to(device)
            accu_num += torch.eq(predict_y, labels.to(device)).sum()
            ##记录到tensorboard
            tensor_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                     accu_loss.item() / (step + 1),
                                                                                     accu_num.item() / sample_num
                                                                                     )
        tb_writer.add_scalar(tags[0], accu_loss.item() / (step + 1), epoch)
        tb_writer.add_scalar(tags[1],  accu_num.item() / sample_num)
        ##记录日志
        logger.info("[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                             accu_loss.item() / (step + 1),
                                                                             accu_num.item() / sample_num
                                                                             ))
    return


def test(model, tensor_loader, criterion, device,tb_writer,tags):
    model.eval()
    test_acc = 0
    test_loss = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return

    
@logger.catch
def main():
    logger.add("./logs/csi_widar3_{time}.log")
    root = r'E:\1-widar\Widar3.0\CSI\20181109_ahnu/'
    #root = r'E:\1-widar\benchmark/'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices = ['UT_HAR_data','NTU-Fi-HumanID','NTU-Fi_HAR','Widar','Widar3'],default='Widar3')
    parser.add_argument('--model', choices = ['MLP','LeNet','ResNet18','ResNet50','ResNet101','RNN','GRU','LSTM','BiLSTM', 'CNN+GRU','ViT'],default='LSTM')
    args = parser.parse_args()

    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root,logger=logger)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tb_writer = SummaryWriter()
    tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
    train(
        model=model,
        tensor_loader= train_loader,
        num_epochs= train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device,
        tb_writer=tb_writer,
        tags=tags,
        logger=logger
         )
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device= device,
        tb_writer = tb_writer,
        tags = tags
        )
    return


if __name__ == "__main__":
    main()
