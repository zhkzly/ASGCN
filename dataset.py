from torch.utils.data import Dataset,DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch


class Mydataset(Dataset):
    '''
    data:   output,list of tuple,[(Xh,Xd,Xw,y)] the Xh is array
    但是相应的label 又是什么呢
    '''
    def __init__(self,data):
        '''

        :param data:dict(Xh,Xd,Xw,y),tensor,
        '''
        super(Mydataset, self).__init__()
        self.data=data
    def __len__(self):
        return self.data['Xh'].shape[0]

    # 所以这个批量是按照什么来呢
    def __getitem__(self,index):
        return self.data['Xh'][index],self.data['Xd'][index],self.data['Xw'][index],self.data['y'][index]


def collate_fn(bn_data):
'''
    this func is used in dataloader,but i found this func was not necessary later ,so i didn't use it in code ,in the begaining ,i wanted to use this to stack the data ,instead of generate the final data directly
'''
    Xh=[]
    Xd=[]
    Xw=[]
    y=[]
    for data in bn_data:
        Xh.append(data[0])
        Xd.append(data[1])
        Xw.append(data[2])
        y.append(data[-1])
    Xh=torch.stack(Xh,dim=0)
    Xd=torch.stack(Xd,dim=0)
    Xw=torch.stack(Xw,dim=0)
    y=torch.stack(y,dim=0)

    return (Xh,Xd,Xw,y)






























