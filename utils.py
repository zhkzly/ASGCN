import torch
import numpy as np
import random
from typing import Optional

from collections import deque

def calculate_gcn(A, X, theta, S: object, K=3):
    '''
    this is used to calculate the graph convolution
    :param A: adjacent matrix:tensor
    :param X,input tensor ,(b,N,C_r_1,T_r_1)，需要转换适用与multi_features and multi_time_step
    也就是使用事件相关性修正过的 X ->X_hat
    :param theta,tensor (K,C_r_1,Cr)
    :param S,tensor (b,N,N) 空间相关性
    :param K: kernel size of the graph convolution
    :return:output:tensor (C_r,N,Tr)    每个时间步产生Cr个，也就是有Cr个filters

   gθ ∗G x = gθ(L)x = K −1 ∑ k=0 θk Tk (  ̃ L)x  某个时间步的一个特征维度的卷积
    突然想起来这些东西在哪计算
    '''
    # Laplacian L
    device = X.device
    A=A.to(device)
    with torch.no_grad():
        rows, cols = A.shape
        Tr = X.shape[-1]
        D = torch.diag(torch.sum(A, dim=1))
        #论文代码中没有使用归一化
        L = D-A
        # print((A==1).any())
        # print(torch.linalg.eig(A))
        lambda_max = torch.max(torch.linalg.eigvals(A).real)
        #print(A==A.T)
        #所以这个地方真的狗，特征值是零
        #print(f'eig of A:{lambda_max}')
        L_hat = 2 * L /(lambda_max+1e-8) - torch.eye(n=rows, m=cols).to(device)
        K, Cr_1, Cr = theta.shape
        N = rows

        # 构造一个
    #print(f'this is S:{S}')
    T_L = [torch.eye(rows, cols).to(device), L_hat[:, :]]  # 进行copy
    assert K > 0
    if K == 1:
        # 也就是车比雪夫多项式的计算不太一样了，实际上，之后是一样的
        return
    elif K == 2:
        return
    else:
        # 计算车比雪夫的T_k
        for i in range(K - 2):
            #使用元素乘法
            #T_ = 2 * torch.matmul(L_hat, T_L[i + 1]) - T_L[i]
            T_ = 2 *L_hat*T_L[i + 1] - T_L[i]
            T_L.append(T_)
        for i, t_ in enumerate(T_L):
            #print(type(i))

            T_L[i] = t_ * S
        #     print(f'this is 车比雪夫：{torch.isnan(T_L[i]).any()}')
        #     print(f'\n this is T_L{i}:{T_L[i]}')
        # print('\n')
    B=X.shape[0]
    result=torch.zeros((B,N,Cr,Tr)).type_as(X)
    for t in range(Tr):
        #(b,N,C_r_1)
        x=X[:,:,:,t]
        output=torch.zeros((B,N,Cr)).type_as(X)
        for k,_ in enumerate(theta):
            #(C_r_1,Cr)
            theta_k=theta[k]
            #(b,N,N)
            tl=T_L[k].type_as(X)
            # (b,V,V)@(b,V,C_r_1)->(b,V,C_r_1)
            rhs=torch.matmul(tl.permute(dims=(0,2,1)), x)
            output=output+torch.matmul(rhs,theta_k)
        result[:,:,:,t]=output
    #print(f'this is  the result of gcn before relu  {result.permute(dims=(0,2,1,3))}')
    return torch.relu(result.permute(dims=(0,2,1,3)))
    # B,N,C_r_1,T_r_1=X.shape
    # K,_,Cr=theta.shape
    # result=torch.zeros((B,Cr,N,T_r_1))
    # result.type_as(X)
    #(X:B,N,C_r_1,T_r_1),如此必不会out of memory
    # for b in range(B):
    #     for tr in range(T_r_1):
    #         for j, cr in enumerate(theta.permute(dims=(2,1,0))):
    #
    #             for i,c_r_1 in enumerate(cr):   #c_r_1
    #                 #(c_r_1:K)
    #                 tempt = []
    #                 for m, k in enumerate(c_r_1):
    #                     tempt.append(T_L[m][b]*k)
    #                 tempt_sum=sum(tempt).type_as(X)
    #                 # print(tempt_sum.shape)
    #                 result[b,j,:,tr]=result[b,j,:,tr]+(torch.matmul(tempt_sum,X[b,:,i,tr].unsqueeze(dim=-1))).squeeze()
           # 使用的是一维的卷积,需要把最后的结果permute以下(b,Cr,N,T_r_1)


# 时间维卷积，需要注意卷积核的大小，步长，padding
#测试图卷积(b,N,C_r_1,T_r_1)
# X=torch.rand(size=(3,4,5,6))
# K=4
# #(K,C_r_1,Cr)
# theta=torch.rand(size=(K,5,7))
# S=torch.rand(size=(3,4,4))
# A=torch.rand(size=(4,4))
# #output:shape(3,7,4,6)
# output=calculate_gcn(A,X,theta,S)
# print(f'the shape of output:{output.shape}')






# calculate adjacent matrix
def calculate_adj_matrix(X):
    '''
    :param X:array,(N,3)->from ,to ,distance
    :return:Y:array,(N,N)
    '''
    # 是否需要排序
    # N=len(sorted(set(X[:,0])))
    # N=len((set(X[:,0])))
    # number2index={num:i for i,num in enumerate(X[:,0])}
    # A=np.zeros((N,N))
    # just like 坐标，x,y,定位
    # for x in X[:,0]:
    #     for y in X[:,1]:
    #         A[number2index[x],number2index[y]]=X[x,2]
    # A=A+A.T #这样就有了两个了
    N = int(np.max(X)+1)
    A = np.zeros((N, N))
    for data in X:
        A[int(data[0]), int(data[1])] = 1
    return A


# 测试邻接矩阵
# adj_data = np.loadtxt(open('data/PEMS04/distance.csv', 'rb'), delimiter=',', skiprows=1, usecols=[0, 1])
# print(f'the first row of the data :{adj_data[0]}')
# print(f'the last row of the data :{adj_data[-1]}')
# print(f'the shape of the data :{adj_data.shape}')
# A=calculate_adj_matrix(adj_data)
# print(f'the shape of the A:{A.shape}')
# print(A[:22,:22])
# print(f'(0,92):{A[0,92]},(1,46):{A[1,46]}')
# #(0,92)=1






# form the dataset (data,label),array :

def generate_data(original_data, Th=24, Td=12, Tw=24, Tp=12, q=288) -> tuple:
    '''
    this  is used to generate dataset from the original data,because the original data is just organized in the form of
    (Tr,N,features),however we need some time-step to predict following time-steps ,so we need to split them into the proper size
    :param original_data ,array,(Tr,N,features),时间步，节点数，特征
    :param Th :int ,用预测时间步前多少个小时的数据，也就是recent segment
    :param Td,int ,用与预测时间步前的多少天的数据，也就是daily-period segment
    :param Tw,int ,用于预测时间片段前的多少周的数据进行，也就是用来产生weekly-period segment
    :param Tp,int ,需要预测的时间片段的长度，也就是最后的y的维度
    :param q,int ,一天有多少个时间步
    :return:output,list of tuple,[(Xh,Xd,Xw,y)] the Xh is array
    (Xh,Xd,Xw,y),分别都是对应的batch
    
    return tuple (Xh,Xd,Xw,y),Xh:(b,T,N,C)
    '''
    # 所以它是如何处理数据边缘的情况
    # 算了，我先按照，边缘舍弃吧

    minum_num = (Tw // Tp) * 7 * q + Tp
    assert minum_num <= original_data.shape[0]
    original_start = Tw // Tp * 7 * q  # 原论文是按照从后往前进行的
    output = []
    list_Xh = []
    list_Xd = []
    list_Xw = []
    list_y = []
    while original_start + Tp <= original_data.shape[0]:
        # 多少小时前的
        Xh = generate_one_data(data=original_data[original_start - (Th // Tp) * 12:original_start], Th=Th, Tp=Tp,
                               multi=1, q=12)
        Xd = generate_one_data(data=original_data[original_start - (Td // Tp) * 1 * q + Tp:original_start + Tp], Th=Td,
                               Tp=Tp, multi=1)
        Xw = generate_one_data(data=original_data[original_start - (Tw // Tp) * 7 * q + Tp:original_start + Tp], Th=Tw,
                               Tp=Tp, multi=7)
        y = original_data[original_start:original_start + Tp]
        # (T,N,C)
        list_Xh.append(torch.from_numpy(Xh))
        list_Xd.append(torch.from_numpy(Xd))
        list_Xw.append(torch.from_numpy(Xw))
        list_y.append(torch.from_numpy(y))
        original_start += 1
    Xh = torch.stack(list_Xh, dim=0)
    Xd = torch.stack(list_Xd, dim=0)
    Xw = torch.stack(list_Xw, dim=0)
    y = torch.stack(list_y, dim=0)

    return (Xh, Xd, Xw, y)


def generate_one_data(data, Th, Tp, multi, q=288):
    '''

    :param data:array,(T,N,features),原有数据的一个截取
    :param Th:
    :param Tp: 需要预测是时间片段
    :param q: 每一天的时间片段量
    :param multi: 判断是一天还是七天
    :return:
    '''
    output = []
    for i in range(Th // Tp):
        output.append(data[i * q * multi:i * q * multi + Tp])
    output = np.concatenate(output, axis=0)

    return output


# 用来测试数据集的大小是否符合要求
# data=np.ones((10000,309,3))
# b=generate_data(data)
# print(b[1].shape)

def transform_data_structure(dataset):
    '''

    将数据的组织形式进行改变，同时将数据类型由array->tensor,这个函数也可以直接放置在generate_data函数中，但是当时没这么考虑
    :param dataset:list of tuple(Xh,Xd,Xw,y)
    :return: (Xh,Xd,Xw,y)
    '''
    list_Xh = []
    list_Xd = []
    list_Xw = []
    list_y = []
    for data in dataset:
        list_Xh.append(torch.from_numpy(data[0]))
        list_Xd.append(torch.from_numpy(data[1]))
        list_Xw.append(torch.from_numpy(data[2]))
        list_y.append(torch.from_numpy(data[-1]))
    Xh = torch.stack(list_Xh, dim=0)
    Xd = torch.stack(list_Xd, dim=0)
    Xw = torch.stack(list_Xw, dim=0)
    y = torch.stack(list_y, dim=0)
    return (Xh, Xd, Xw, y)


def dataiter(dataset, batch_size, sampling=True):
    '''
    用来产生训练所需的batch，data,but i found this func was not necessary
    :param dataset: (Xh,Xd,Xw,y);Xh:(b,Xh),tensor
    :param batch_size:
    :param sampling:
    :return:
    '''
    num_of_dataset = dataset[0].shape[0]

    index = np.arange(num_of_dataset)
    if sampling:
        np.random.shuffle(index)
    assert batch_size < num_of_dataset
    Xh, Xd, Xw, y = dataset
    for i in range(num_of_dataset // batch_size):
        yield (Xh[index[i * batch_size:(i + 1) * batch_size]], Xd[index[i * batch_size:(i + 1) * batch_size]], \
               Xw[index[i * batch_size:(i + 1) * batch_size]], y[index[i * batch_size:(i + 1) * batch_size]])


def generate_train_val_test_set(dataset: tuple, ratio=(6, 8, 10), shuffle=False, merge=False):
    '''
    用来产生所需要的数据集
    :param dataset: tuple (tensor,tensor,tensor),也就是return of generate_dataset  #(num_of_data,T,N,C)
    :param ratio: 划分的比例，
    :param shuffle:boolean,数据项是否随机的选取放置在相应的train_set,val_set,test_set中
    :param merge: 是否需要val_set,if len(ratio)==2,也就是相当于merge==True
    :return:dataset:(train_set,val_set,test_set),train_set:tuple(num_of_dataset,Xh,Xd,Xw,y)
    '''
    # print(f'shape of dataset:{dataset[0].shape}')
    num_of_data = dataset[0].shape[0]
    # if len(ratio) == 3 and merge:
    #     print('1')
    # ~Fasle==-2,so this code is wrong
    # if len(ratio) == 2 and ~merge:
    #     print('2')
    if len(ratio) == 3 and merge or len(ratio) == 2 and not merge:
        raise ValueError('the shape of ratio and the value of the merge are not consistent')
    if not merge:
        if shuffle:
            index_ = np.arange(num_of_data)
            np.random.shuffle(index_)
            train_set = [data[index_[0:int(num_of_data * ratio[0] / 10)]] for data in dataset]
            val_set = [data[index_[int(num_of_data * ratio[0] / 10):int((num_of_data * ratio[1]) / 10)]] for data in
                       dataset]
            test_set = [data[index_[int(num_of_data * ratio[1] / 10):]] for data in dataset]
        else:
            index_ = np.arange(dataset[0].shape[0])
            train_set = [data[index_[0:int(num_of_data * ratio[0] / 10)]] for data in dataset]
            val_set = [data[index_[int(num_of_data * ratio[0] / 10):int((num_of_data * ratio[1]) / 10)]] for data in
                       dataset]
            test_set = [data[index_[int(num_of_data * ratio[1] / 10):]] for data in dataset]
        print(
            f'train_set:Xh:{train_set[0].shape},Xd:{train_set[1].shape},Xw:{train_set[2].shape},y:{train_set[3].shape}')
        print(f'val_set:Xh:{val_set[0].shape},Xd:{val_set[1].shape},Xw:{val_set[2].shape},y:{val_set[3].shape}')
        print(f'test_set:Xh:{test_set[0].shape},Xd:{test_set[1].shape},Xw:{test_set[2].shape},y:{test_set[3].shape}')
        return train_set, val_set, test_set
    else:
        if shuffle:
            index_ = np.arange(dataset[0].shape[0])
            np.random.shuffle(index_)
            train_set = [data[index_[0:int(num_of_data * ratio[0] / 10)]] for data in dataset]
            test_set = [data[index_[int(num_of_data * ratio[1] / 10):]] for data in dataset]
        else:
            index_ = np.arange(dataset[0].shape[0])
            train_set = [data[index_[0:int(num_of_data * ratio[0] / 10)]] for data in dataset]
            test_set = [data[index_[int(num_of_data * ratio[1] / 10):]] for data in dataset]
        print(
            f"train_set:Xh:{train_set[0].shape},Xd:{train_set[1].shape},Xw:{train_set[2].shape},y:{train_set[3].shape}")
        print(f"test_set:Xh:{test_set[0].shape},Xd:{test_set[1].shape},Xw:{test_set[2].shape},y:{test_set[3].shape}")
        return train_set, test_set


# 测试数据集划分的正确与否
# data=np.ones((5000,100,3))
# b=generate_data(data)
# dataset=generate_train_val_test_set(b,shuffle=True,ratio=(5,5),merge=True)


def normalization(data: tuple, num_of_dataset_type):
    '''

    :param data: tuple of (train_set,val_set,test_set),b,t,n,c
    :return: dict of (mean,std),用来进行预测时使用，train_norm_set,val_norm_set,test_norm_set
    '''
    # with torch.no_grad:   本来就是requires_grad=False,所以不需要,原论文是使用各自的数据进行的归一化
    #means, stds = [], []
    #for i, train_data_X in enumerate(data[0]):
        #if i == 3:
            #break
        #means.append(torch.mean(train_data_X, dim=(0,1,2), keepdim=True))
        #stds.append(torch.std(train_data_X, dim=(0,1,2), keepdim=True))
    data_tempt=torch.cat(data[0][:-1],dim=1)  #b,T'N,C,in the paper ,the data is normalized along the channels,we can cat them along the time_step,
    mean=torch.mean(data_tempt,dim=(0,1,2),keepdim=True)
    std=torch.std(data_tempt,dim=(0,1,2),keepdim=True)
    data_normal = [[] for i in range(num_of_dataset_type)]
    for j, data_type in enumerate(data):
        for i, X in enumerate(data_type):
            if i == 3:
                data_normal[j].append(X)
            else:
                data_normal[j].append((X - mean) /(std))
    state = {'mean': mean, 'std': std}
    return state, data_normal


# #测试normalization
# data=np.ones((5000,100,3))
# b=generate_data(data)
# dataset=generate_train_val_test_set(b)
# state,data_norm=normalization(dataset,3)
# print(f"state means :{state['means'][0].shape},state stds :{state['stds'][0].shape} data_norm:{data_norm[0][0].shape} ")


def data_preprocession(data_file_path, Th=24, Td=12, Tw=24, Tp=12, num_per_hour=12, merge=False, dtype=np.float32):
    '''
    用来产生所需要的数据集，并将其转换为我们所需要的形式
    :param data_file_path: where to load data
    :param Th: 可以用来确定多少小时（Th//Tp）也就是多少个小时，一切按照论文来了
    :param Td: 可以用来确定多少天
    :param Tw: 可以确定多少周
    :param Tp: 预测的时间长度
    :param num_per_hour:
    :param merge: 是否将train_set,val_set 进行融合
    :return: dict，包含所有的数据集，所需要的均值（用作predict）,Xh(B,T,N,C)
    '''

    # 首先，文章中的训练集，交叉验证集，测试集都是按照时间直接选取的，没有随机的去选择，我选择随机抽取，

    original_dataset = np.load(data_file_path)['data']

    q = 24 * num_per_hour  # 一天多少个sample,注意进行简单的测试的时候，最好选择original_dataset[0:5000]，否则内存可能会炸，8个多G
    dataset: tuple = generate_data(original_dataset.astype(dtype), Th=Th, Td=Td, Tw=Tw, Tp=Tp, q=q)
    before_normalization = generate_train_val_test_set(dataset, shuffle=False)
    # 归一化
    stats, norm_set = normalization(data=before_normalization, num_of_dataset_type=3)
    # 将数据转换为dict,方便操作，否则用索引太麻烦了，容易忘
    stats_means = stats['mean']
    stats_stds = stats['std']
    # 其实这里应该判断是否需要merge
    if not merge:
        norm_dataset = {
            'stats':
                {
                    'mean': stats_means,
                    'std': stats_stds
                },
            'train_norm_set':
                {
                    'Xh': norm_set[0][0],
                    'Xd': norm_set[0][1],
                    'Xw': norm_set[0][2],
                    'y': norm_set[0][3][:,:,:,0]
                },
            'val_norm_set':
                {
                    'Xh': norm_set[1][0],
                    'Xd': norm_set[1][1],
                    'Xw': norm_set[1][2],
                    'y': norm_set[1][3][:,:,:,0]
                },
            'test_norm_set':
                {
                    'Xh': norm_set[2][0],
                    'Xd': norm_set[2][1],
                    'Xw': norm_set[2][2],
                    'y': norm_set[2][3][:,:,:,0]
                }

        }
    else:
        norm_dataset = {
            'stats':
                {
                    'means': stats_means,
                    'stds': stats_stds
                },
            'train_norm_set':
                {
                    'Xh': norm_set[0][0],
                    'Xd': norm_set[0][1],
                    'Xw': norm_set[0][2],
                    'y': norm_set[0][3][:,:,:,0]
                },
            'test_norm_set':
                {
                    'Xh': norm_set[2][0],
                    'Xd': norm_set[2][1],
                    'Xw': norm_set[2][2],
                    'y': norm_set[2][3][:,:,:,0]
                }

        }

    return norm_dataset


# data_preprocession 的测试
# norm_dataset=data_preprocession(data_file_path='data/PEMS04/pems04.npz')
# print(f"norm train_set Xh:{norm_dataset['train_norm_set']['Xh'].shape},"
#       f"norm_means_Xh:{norm_dataset['stats']['means']['Xh'].shape} ")

from dataset import Mydataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm


# 测试数据集的加载
# norm_dataset=data_preprocession(data_file_path='data/PEMS04/pems04.npz')
# train_dataset=Mydataset(norm_dataset['train_norm_set'])
# dataloader=DataLoader(train_dataset,batch_size=13)
# pbar=tqdm(dataloader,total=len(dataloader))
#
# for i,(Xh,Xd,Xw,y) in enumerate(pbar):
#     print(f'Xh shape:{Xh.shape},y shape:{y.shape}')
#     if i ==10:
#         break


def get_configs( num_of_nodes, Th, Td, Tw, Tp=12,  num_of_time_channels=64,PEMS04=True,
                out_channels=64,
                in_channels=3, gcn_conv_kernel_size=(1, 3), gcn_conv_padding=(0, 1), \
                K=3, dtype=torch.float32):
    configs = {}
    config_h = {
        'num_of_nodes': num_of_nodes,
        'in_channels': in_channels,
        'time_step': int(Th),
        'num_of_time_channels': num_of_time_channels,  # 默认64
        'gcn_conv_kernel_size': gcn_conv_kernel_size,  # (1,3)
        'gcn_conv_stride': (1, Th // Tp),  # num_of_weeks(1,num_of_weeks)
        'gcn_conv_padding': gcn_conv_padding,  # (0,1)
        'K': K,
        'out_channels': out_channels,
        'dtype': dtype
    }
    config_d = {'num_of_nodes': num_of_nodes,
                'in_channels': in_channels,
                'time_step': int(Td),
                'num_of_time_channels': num_of_time_channels,  # 默认64
                'gcn_conv_kernel_size': gcn_conv_kernel_size,  # (1,3)
                'gcn_conv_stride': (1, Td // Tp),  # num_of_weeks(1,num_of_weeks)
                'gcn_conv_padding': gcn_conv_padding,  # (0,1)
                'K': K,
                'out_channels': out_channels,
                'dtype': dtype}
    config_w = {'num_of_nodes': num_of_nodes,
                'in_channels': in_channels,
                'time_step': int(Tw),
                'num_of_time_channels': num_of_time_channels,  # 默认64
                'gcn_conv_kernel_size': gcn_conv_kernel_size,  # (1,3)
                'gcn_conv_stride': (1, Tw // Tp),  # num_of_weeks(1,num_of_weeks)
                'gcn_conv_padding': gcn_conv_padding,  # (0,1)
                'K': K,
                'out_channels': out_channels,
                'dtype': dtype}
    Xh = [config_h]
    Xd = [config_d]
    Xw = [config_w]
    config2 = {
        'num_of_nodes': num_of_nodes,
        'in_channels': 64,
        'num_of_time_channels': 64,
        'time_step':Tp,
        'gcn_conv_kernel_size': gcn_conv_kernel_size,
        'gcn_conv_stride': (1,1),
        'gcn_conv_padding': gcn_conv_padding,
        'num_of_prediction':Tp,
        'K': K,
        'out_channels': out_channels,
        'dtype': dtype
    }
    Xh.append(config2)
    Xd.append(config2)
    Xw.append(config2)
    configs['h'] = Xh
    configs['d'] = Xd
    configs['w'] = Xw

    return configs


#测试get_configs()



from sklearn.metrics import mean_absolute_error,mean_squared_error




def masked_mse(y_hat,y_label,null_value=torch.nan):
    '''
    this is used to calculate teh root mean square error,the meaning of the mask is to mask the invalid values
    :param y_hat: tensor,(b,T,N)
    :param y_label: (b,T,N)
    :return: scalar
    '''
    with torch.no_grad():
        if torch.isnan(torch.tensor([null_value])).any():
            mask= not torch.isnan(y_label)
        else :
            mask=(y_label!=null_value)

        mask=mask.type_as(y_label)
        mask/=mask.mean()
        mask=torch.where(torch.isnan(mask),torch.zeros_like(mask),mask)
        #torch.nan_to_num()
        loss=torch.pow((y_hat-y_label),exponent=2)*mask
        #计算指数后也有可能是nan
        loss=torch.mean(torch.where(torch.isnan(loss),torch.zeros_like(mask),loss))
    return loss.item()

def masked_rmse(y_hat,y_label,null_value=torch.nan):
    '''
    calculate the rmse
    :param y_hat:
    :param y_label:
    :param null_value:
    :return:
    '''
    return torch.sqrt(masked_mse(y_hat,y_label,null_value))

def masked_mae(y_hat,y_label,null_value=torch.nan):
    with torch.no_grad():
        if torch.isnan(torch.tensor([null_value])).any():
            mask= not torch.isnan(y_label)
        else :
            mask=(y_label!=null_value)

        mask=mask.type_as(y_label)
        mask/=mask.mean()
        mask=torch.where(torch.isnan(mask),torch.zeros_like(mask),mask)
        #torch.nan_to_num()
        loss=torch.abs((y_hat-y_label))*mask
        #计算指数后也有可能是nan
        loss=torch.mean(torch.where(torch.isnan(loss),torch.zeros_like(mask),loss))
    return loss.item()

def masked_mape(y_hat,y_label,null_value=torch.nan):
    '''
    calculate the mape error
    :param y_hat:
    :param y_label:
    :param null_value:
    :return:
    '''
    with torch.no_grad():
        if torch.isnan(torch.tensor([null_value])).any():
            mask = not torch.isnan(y_label)
        else:
            mask = (y_label != null_value)
        mask=mask.type_as(y_hat)
        mask/=mask.mean()
        mape=torch.abs((y_label-y_hat)/y_label)
        mape=torch.nan_to_num_(mape*mask)
        mape=torch.mean(mape).item()
        return mape



def re_normalization(x,_mean,_std):
    '''
    this is used to recover the original data
    :param x: (b,T,N,C)
    :param _mean: (1,1,1,3)
    :param _std: (1,1,1,3)
    :return:
    '''
    with torch.no_grad():
        x=x*_std+_mean

        return x.detach().numpy()


def save_data_to_txt(path_to_save,data):
    '''
        this is used to save the data to the txt so we can load it in the feature
        
        :params:data [[],[]]
    '''
    with open(path_to_save,'a') as f:
        for numbers in data:
            if isinstance(numbers,(list,tuple,deque)):
                for number in numbers:
                    f.write(str(number))
                f.write(';')
                
            else:
                raise ValueError('we need the list ')


def read_data_saved_to_txt(path_to_open):
    '''
    read the data return from the save_data_to_txt function 
    :return :[[],[]]
    '''



def evaluate(model,dataset,_mean,_std,model_params_path,inputs_save_path='input_save_path',outputs_save_path='outputs_save_path',dtype=torch.float32,batch_size=16):
    '''
    用来评估模型的运行结果
    :param model:
    :param dataset:
    :param _mean:
    :param _std:
    :param save_result_to:
    :param model_params_path:
    :return:
    '''
    with torch.no_grad():
        net=model
        net.eval()
        net.load_state_dict(torch.load(model_params_path))
        dataset=Mydataset(dataset)
        dataloader=DataLoader(dataset,batch_size=batch_size)
        pbar=tqdm(dataloader,total=len(dataloader))
        pbar.set_description('this is used to evaluate the model')
        net=net.cpu()

        y_hat=deque()
        input=deque()
        y_label=deque()
        for xh,xd,xw,y in pbar:
            xh=xh.cpu()
            xd=xd.cpu()
            xw=xw.cpu()
            y=y.cpu()
            #print(f'xh.shape:{xh.shape}')
            y_pred=net([xh,xd,xw])
            y_hat.append(y_pred.detach().numpy())
            y_label.append(y.detach().numpy())
            input.extend([re_normalization(x,_mean,_std) for x in [xh,xd,xw]])
            pbar.set_description(f'xh.shape:{xh.shape},y.shape:{y_pred.shape}')
        #(xh,xd,xw),xh:(B,T,N,C)
        #inputs=deque([np.stack([*k],axis=0) for k in zip(*input)]),批量不足可能会导致不同的大小
        np.savez(inputs_save_path,*input)
        np.savez(outputs_save_path,*y_hat)

        y_label=np.concatenate(y_label,axis=0)
        outputs=np.concatenate(y_hat,axis=0)

        rmes=np.sqrt(mean_squared_error(y_label.reshape(-1,1),outputs.reshape(-1,1)))
        mae=mean_absolute_error(y_label.reshape(-1,1),outputs.reshape(-1,1))
        mes=mean_squared_error(y_label.reshape(-1,1),outputs.reshape(-1,1))
        np.savez('result',rmes=rmes,mae=mae,mes=mes)

        return rmes,mes,mae
