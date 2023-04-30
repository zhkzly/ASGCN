import torch
import numpy as np
from dataset import Mydataset
from torch.utils.data import DataLoader
import utils
from dataset import collate_fn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt


def Trainer(model, data, optimizer, save_params_to='params_of_model',loss_history_path='loss_history_path', writer = None, loss=torch.nn.MSELoss(),
            epochs=100, batch_size=16,device=('cuda' if torch.cuda.is_available() else 'cpu'), shuffle=False, distribute=False):
    '''
    用来进行模型的训练
    :param distribute:
    :param shuffle: 是否对dataloader中的数据进行打乱
    :param device: torch.device
    :param model:nn
    :param dataset:tuple (Xh,Xd,Xw,y),tensor,return of the generate_dataset()
    :param optimizer:torch.optim
    :param loss:such as torch.nn.MSEloss
    :param epochs:
    :param batch_size:
    :return:
    '''

    dataset = Mydataset(data)
    # optimizer.to(device)
    # loss.to(device)
    if distribute:
        dist.init_process_group(backend='nccl')
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                sampler=sampler)

        local_rank = int(os.environ['LOCAL_RANK'])
        model.to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        loss.to(device)
        l_=0.0
    else:
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        model.to(device)

        loss.to(device)
        l_ = 0.0

    #pbar= tqdm(enumerate(dataloader),desc='training_procession',total=len(dataloader))
    history_loss=[]
    #print(epochs)
    for epoch in range(epochs):
        if distribute:
            sampler.set_epoch(epoch)    #这里有问题就是因为可能没有事先声明
       # else:
           # pass
        loss_his =deque()
        pbar= tqdm(enumerate(dataloader),desc='training_procession',total=len(dataloader))
        for it,batch_set in pbar:
            Xh,Xd,Xw,y=batch_set
            #print(device)
            Xh=Xh.to(device)
            #print(f'Xh which device:{Xh.device}')
            #print(f"this is Xh:{Xh}")
            Xd=Xd.to(device)
            Xw=Xw.to(device)
            y=y.to(device)

            y_hat = model((Xh, Xd, Xw))
            y=y.type_as(y_hat)
            #print(f'this is y lable:{y},\nthis is y_hat:{y_hat}')
            l = loss(y_hat, y)
            l_ = l.item()
            optimizer.zero_grad()
            #print(l)
            #with torch.autograd.detect_anomaly():
            l.backward()
            optimizer.step()
            # print(l.item())
            # assert False
            if distribute:
            # 保存模型的参数，训练过程中的数据loss等的保存，
                if dist.get_rank()==0:
                        pbar.set_description(desc=f"epoch:{epoch},iter:{it},training_loss:{l.item():.5f},lr:{optimizer.param_groups[0]['lr']}")
                        loss_his.append(l_)
                elif dist.get_rank() == 0 and writer is not None:
                    writer.add_scalar('loss', scalar_value=l_, global_step=50)
            else:
                  pbar.set_description(desc=f"epoch:{epoch},iter:{it},training_loss:{l.item():.5f},lr:{optimizer.param_groups[0]['lr']}")
                  loss_his.append(l_)
        if distribute:
            if dist.get_rank() == 0:
                history_loss.append(loss_his)
        else:
            history_loss.append(loss_his)
            #print('发生了什么')
    if distribute :
        if dist.get_rank()==0:
            torch.save(model.module.state_dict() if hasattr(model,'module') else model.state_dict(),save_params_to)
            loss_array=np.array(history_loss)
            #print(loss_array)
            np.save(file=loss_history_path,arr=loss_array)
            plt.plot(loss_array[:,1])
            plt.ylabel('loss')
            plt.xlabel('iter')
            plt.savefig('loss.jpg')
            plt.show()
            return loss_array
    else :
        print('why')
        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), save_params_to)
        loss_array = np.array(history_loss)
        #print(loss_array)
        plt.plot(loss_history_array[:,1])
        plt.ylabel('loss')
        plt.xlabel('iter')
        plt.savefig('loss.jpg')
        plt.show()
        np.save(file=loss_history_path,arr=loss_array)
        return loss_array