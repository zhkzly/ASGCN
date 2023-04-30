import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import calculate_gcn,get_configs
from collections import deque


# TODO:class Residual ,check out the input of all the layer,

#得到的经验是，在书写模型的时候需要考虑模型的批量化计算，每个传入参数的获取如何进行？也就是彼此间的传参

#  note:the shape of the input of the model is (b,T,N,C)->(batch_size,time_step,num_of_nodes,features or channels),in addition ,the test part of this .py may be wrong ,
# for i have changed  some layers of the model after check the module,


class SAtt(nn.Module):
    def __init__(self,num_of_nodes:int,in_channels:int,time_step:int,dtype=torch.float32):
        super(SAtt,self).__init__()
        # this can be change in 
        #for params in self.parameters
        #if len(params.shape)>=2 nn.init.xavier_uniform.....
        self.W1=torch.nn.Parameter(torch.empty((time_step,),dtype=dtype))
        torch.nn.init.xavier_uniform_(self.W1.view(-1,1))
        self.W2=torch.nn.Parameter(torch.empty((in_channels,time_step),dtype=dtype))
        torch.nn.init.xavier_normal_(self.W2)
        self.W3=torch.nn.Parameter(torch.empty((in_channels,),dtype=dtype))
        torch.nn.init.xavier_uniform_(self.W3.view(-1,1))
        self.b=torch.nn.Parameter(torch.empty((num_of_nodes,num_of_nodes),dtype=dtype))
        torch.nn.init.xavier_uniform_(self.b)
        self.Vs=torch.nn.Parameter(torch.empty((num_of_nodes,num_of_nodes),dtype=dtype))
        self.dtype=dtype

    def forward(self,X):
        '''
        #输入的数据其实是(b,T,N,C):但是在写模型的时候，完全按照paper了,finally ,i change the input before passing them into model layer
        :param X: tensor,(b,N,C,T),N:the number of the nodes,C,features,T,time_step,(b,N,C,T)
        :return: tensor,(b,N,N)
        '''
        X=X.type(self.dtype)
        #print(f'x which device:{X.device}')
        #print(torch.matmul(X,self.W1.view(-1,1)).squeeze().shape)->(b,N,C)@(C,T)->(b,N,T)
        output1=torch.matmul(torch.matmul(X,self.W1.view(-1,1)).squeeze(),self.W2)
        #print(output1.shape)   #(b,T,N,C),W3(C,1)->(b,T,N)
        output=torch.matmul(X.permute(dims=(0,3,1,2)),self.W3.view(-1,1)).squeeze()   #让高维数的放在前面
        output=torch.matmul(output1,output)+self.b.unsqueeze(dim=0)  #(b,N,N)
        S=torch.matmul(self.Vs.unsqueeze(dim=0),torch.sigmoid(output))#(b,N,N)
        #S=S-torch.max(S,dim=2,keepdim=True)[0]
        S=torch.softmax(S,dim=1)    #按照列，右乘的时候加权和就是一了
        #S=torch.exp(S)/torch.sum(S,dim=2,keepdim=True)
        #print(f'this is SAtt S:{S}\n\n')

        return S
#测试SAtt
# a=SAtt(3,4,5)
# X=torch.rand((3,3,4,5))
# b=a(X)
# print(f'the shape of output:{b.shape},the values of output:{b}')
# #(output :3:b,3:num_of_nodes,3:num_of_nodes)

class TAtt(nn.Module):
    def __init__(self,num_of_nodes:int,in_channels:int,time_step:int,dtype=torch.float32):
        super(TAtt,self).__init__()
        self.U1=nn.Parameter(torch.empty((num_of_nodes,),dtype=dtype))
        torch.nn.init.xavier_uniform_(self.U1.view(-1,1))
        self.U2=nn.Parameter(torch.empty((in_channels,num_of_nodes),dtype=dtype))
        torch.nn.init.xavier_uniform_(self.U2.view((-1,1)))
        self.U3=nn.Parameter(torch.empty((in_channels,),dtype=dtype))
        torch.nn.init.xavier_uniform_(self.U3.view(-1,1))
        self.be=nn.Parameter(torch.empty((time_step,time_step),dtype=dtype))
        torch.nn.init.xavier_uniform_(self.be)
        self.Ve=nn.Parameter(torch.empty((time_step,time_step),dtype=dtype))
        torch.nn.init.xavier_uniform_(self.Ve)
        self.dtype=dtype

    def forward(self,X):
        '''
        用来进行时间维度的attention，需要注意的是这个是个批量操作
        :param X: tensor,(b,N,C,T)->这样搞完全是因为paper的公式，我完全按照他来了
        :return:
        '''
        X=X.type(self.dtype)
        #(b,T,C,N)@(N,1)->(b,T,C)@(C,N)->(b,T,N)
        output1=torch.matmul(torch.matmul(X.permute(dims=(0,3,2,1)),self.U1.view(-1,1)).squeeze(),self.U2)
        #(b,T,N)@[(b,T,N,C)@(C,1)]->(b,T,T)
        output=torch.matmul(output1,torch.matmul(X.permute(dims=(0,3,1,2)),self.U3.view(-1,1)).squeeze().permute(dims=(0,2,1)))+self.be.unsqueeze(dim=0)

        E=torch.matmul(self.Ve.unsqueeze(dim=0),torch.sigmoid(output))
        E=torch.softmax(E,dim=1)    #好像都是用的列，之后右乘可以是加权和
        #(b,N,C,T)@(b,T,T)->(b,N,C,T)
        X_hat=torch.matmul(X,E.unsqueeze(dim=1))
        #print(f'this is X_hat of TAtt:{X_hat}\n\n')
        return X_hat



#测试TAtt
# a=TAtt(3,4,5)
# #(b,N,C,T)
# X=torch.rand((3,3,4,5))
# b=a(X)
# print(f'the shape of output:{b.shape},the values of output:{b}')
#E(3,5,5),(X_hat:3,3,4,5)   X_hat=X*E

# 空间卷积操作
# 图卷积：graph convolution
#超参数：图卷积核的大小：K
#常规卷积核的大小，时间步，padding等
# formlar

class GCN_conv(nn.Module):
    def __init__(self,A,K:int,in_channels:int ,out_channels:int,num_of_time_channels,stride,kernel_size=(1,3),padding=(0,1),dtype=torch.float32):
        '''
        :param A:邻接矩阵
        :param S:the dependence of each nodes      (b,N,N)
        :param K:考虑周边的K个节点，也即切比雪夫不等式的K
        :param out_channels:C_r,图卷积后，对于每个node,每个时间步产生多少个特征
        :param out_time_step:Tr,这个本质上没啥用，我需要的是kernel
        :param kernel_size:时间维度的standard convolution
        :param padding:in the paper the padding,stride ,kernerl_size is selected on purpose
        :param stride:
        :param dtype:
        '''
        super(GCN_conv,self).__init__()
        self.theta=nn.Parameter(torch.empty((K,in_channels,out_channels),dtype=dtype))
        nn.init.xavier_uniform_(self.theta.view(-1,1))
        #用来进行时间维度的卷积，最终的输出结果会影响，时间步的长度，也就是普通卷积操作的计算方法一致
        self.conv2d1=nn.Conv2d(out_channels, num_of_time_channels,kernel_size=kernel_size, stride=stride, padding=padding, dtype=dtype) #out_channel:C_r
        self.A=A
        self.relu1=nn.ReLU()
        #self.relu2=nn.ReLU()
        self.dtype=dtype


    def forward(self,X,S):
        '''
        用来计算数据的图卷积以及普通卷积的结果，
        :param X:tensor (b,N,Cr_1,Tr_1)
        :return: y:tensor (b,Cr,N,Tr)，Cr,Tr 都是通过卷积来控制的
        '''
        #calculate graph convolution：(b,N,Cr,Tr_1)
        X=X.type(self.dtype)
        gcn=calculate_gcn(self.A,X,self.theta,S)
        # calculate standard convolution  (b,C_r,N,T_r_1)     放置成了这个，然后使用二维卷积
        #print(f'this is the ouput of the gcn calculate_gcn:{gcn}')
        result=self.conv2d1(self.relu1(gcn))
        #print(f'this is result of GCN_conv:{result}\n\n')

        return result

# #测试图卷积和普通卷积(GCN_conv)
# X=torch.rand(size=(3,4,5,6))
# K=4
# #(K,C_r_1,Cr)
# theta=torch.rand(size=(K,5,7))
# S=torch.rand(size=(3,4,4))
# A=torch.rand(size=(4,4))
# #output:shape(3,7,4,6)  #(b,C_r,N,T_r_1)
# output=calculate_gcn(A,X,theta,S)
# print(f'the shape of output:{output.shape}')
#
#
# gcn_conv=GCN_conv(A=A,K=K,in_channels=5,out_channels=7,num_of_time_channels=8,stride=(1,1),kernel_size=(1,1),padding=(0,0))
# result=gcn_conv(X,S)
# #shape:(3,8,4,6)
# print(f'the shape of result:{result.shape}')









# TODO:STblock

class STblock(nn.Module):
    #def __init__(self,num_of_nodes,in_channels,out_channels,time_step,kernel_size,padding,stride,\
     #            A,K,dtype=torch.float32):
    def __init__(self,config:dict,A):
        '''

        :param config:
        '''
        super(STblock,self).__init__()
        self.num_of_nodes=config['num_of_nodes']
        self.in_channels=config['in_channels']
        self.time_step=config['time_step']
        self.out_channels=config['out_channels']
        #this used to calculate the time convolution
        self.gcn_conv_kernel_size=config.setdefault('gcn_conv_kernel_size',(1,3))
        self.gcn_conv_padding=config.setdefault('gcn_conv_padding',(0,1))
        self.gcn_conv_stride=config['gcn_conv_stride']
        self.dtype=config.setdefault('dtype',torch.float32)
        self.A=A.type(self.dtype)
        self.K=config.setdefault('K',3)
        self.num_of_time_channels=config['num_of_time_channels']
        #num_of_nodes:int,in_channels:int,time_step:int,dtype=torch.float32
        self.SAtt=SAtt(self.num_of_nodes,self.in_channels,self.time_step,self.dtype)
        #TAtt:num_of_nodes:int,in_channels:int,time_step:int,dtype=torch.float32
        self.TAtt=TAtt(self.num_of_nodes,self.in_channels,self.time_step,self.dtype)
        #GCN_conv:A,S,K:int,in_channel:int ,out_channels:int,out_time_step:int,kernel_size,padding,stride,dtype=torch.float32
        self.GCN=GCN_conv(self.A,self.K,self.in_channels,self.out_channels,self.num_of_time_channels,kernel_size=self.gcn_conv_kernel_size,\
                          padding=self.gcn_conv_padding,stride=self.gcn_conv_stride,dtype=self.dtype)



    def forward(self,X):
        '''
        每一个计算块
        :param X:(b,N,Cr,Tr)        #实际拥有的数据集是(b,T,N,C)
        :return:(b,Cr,N,tr)
        '''
        X.type(self.dtype)
        output=self.TAtt(X)
        #print(f'this is input of SAtt:{output}')
        S=self.SAtt(output)
 
        output=self.GCN(X,S)
        #print(f'this is the input of next layer:{output}')
        #print(f'this is ouput of S:{S}\n\n')

        return output

# X=torch.rand(size=(3,4,5,6))
# K=4
# #(K,C_r_1,Cr)
# theta=torch.rand(size=(K,5,7))
# S=torch.rand(size=(3,4,4))
# A=torch.rand(size=(4,4))
# #test STblock
# config={'num_of_nodes':4,'in_channels':5,'out_channels':7,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
#         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':8,'time_step':6
#         }
# stblock=STblock(config)
# result=stblock(X)
# #the shape of result=(3,8,4,6)->(b,Cr,N,Tr)
# print(f'the shape of result:{result.shape}')




#TODO:Residual
class Resdiual(nn.Module):
    def __init__(self,config,A):
        super(Resdiual,self).__init__()
        #params=[value for value in config.values]
        self.stblock=STblock(config,A)    #(b,C,N,T)
        self.dtype=config['dtype']
        self.residual=nn.Conv2d(in_channels=config['in_channels'],out_channels=config['num_of_time_channels'],padding=config['gcn_conv_padding']
                                ,kernel_size=config['gcn_conv_kernel_size'],stride=config['gcn_conv_stride'],dtype=self.dtype)
        self.ln=nn.LayerNorm(config['num_of_time_channels'],dtype=self.dtype)
        self.relu=nn.ReLU()
    def forward(self,X):
        '''
            需要保证最后能有相同的大小  ，paper author 使用了卷积核为一的二维卷积调整了大小
            :param X:tensor (b,N,Cr_1,Tr_1)
            return (b,N,Cr,Tr)
        '''
        output=self.stblock(X)  #(b,Cr,N,Tr)
        X=X.type(self.dtype)
        residual= self.residual(X.permute(dims=(0,2,1,3)))  #(b,Cr,N,Tr)->(b,C,N,T)
        #不能使用in-place 否则不能反向传播  (b,Tr,N,C)
        output=output+residual #反正能够知道的就是我的X的大小什么绝对有问题，只能先把形式搭出来吧
        #dims=0,3,2,1(b,T,N,C)->(b,N,C,T)(最终的预测结果应该是这个，需要控制kernel_size)
        #print(f'this is output of Residual:{output}\n\n')
        return self.ln(self.relu(output).permute(dims=(0,3,2,1))).permute(dims=(0,2,3,1))#保证可以串联起来
# X=torch.rand(size=(3,4,5,6))
# K=4
# #(K,C_r_1,Cr)
# theta=torch.rand(size=(K,5,7))
# S=torch.rand(size=(3,4,4))
# A=torch.rand(size=(4,4))
# #test STblock
# config={'num_of_nodes':4,'in_channels':5,'out_channels':7,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
#         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':8,'time_step':6
#         }
# residual=Resdiual(config)
# result=residual(X)
# #the shape of result=(3,4,8,6)->(b,Cr,N,Tr)
# print(f'the shape of result:{result.shape}')




class ASGCN_module(nn.Module):
    def __init__(self,configs,A):
        '''
        用来产生ASGCN中的recent，hours,week 等，这些是ST模块的串行组合，需要输入所有的ST所需要的参数
        :param configs:list(dict)
        {num_of_nodes:N,in_channels:C_{r-1},out_channels:Cr:,time_step:Tr,kernel_size,padding,stride,\
                 A:adja,K,dtype=torch.float32}
        '''
        super(ASGCN_module,self).__init__()
        self.sequential=nn.Sequential(*[Resdiual(config,A) for config in configs])
        self.dtype=configs[-1]['dtype']
        self.final_conv=nn.Conv2d(in_channels=configs[-1]['time_step'],out_channels=configs[-1]['num_of_prediction']
                                  ,kernel_size=(1,configs[-1]['num_of_time_channels']),dtype=self.dtype)

    def forward(self,X):
        '''

        :param X:(b,N,Cr_1,Tr_1)
        :return:(b,N,Cr,Tr)
        '''
        #with torch.autograd.detect_anomaly():
        output= self.sequential(X)
        output=self.final_conv(output.permute(dims=(0,3,1,2))).squeeze().permute(dims=(0,2,1))
        #print(f'this is output of module:{output}\n\n')
        return output

#测试ASGCN_module
# X=torch.rand(size=(3,4,5,6))
# K=4
# #(K,C_r_1,Cr)
# theta=torch.rand(size=(K,5,7))
# S=torch.rand(size=(3,4,4))
# A=torch.rand(size=(4,4))
# #test STblock
# config1={'num_of_nodes':4,'in_channels':5,'out_channels':7,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
#         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':8,'time_step':6
#         }
# #(3,4,8,6)->b,N,C,T
# config2={'num_of_nodes':4,'in_channels':8,'out_channels':9,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
#         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':10,'time_step':6
#         }
# configs=[config1,config2]
# asgcn_module=ASGCN_module(configs)
# result=asgcn_module(X)
#
# #the shape of result=(3,4,8,6)->(3,4,10,6)->(b,N,C,Tr)
# print(f'the shape of result:{result.shape}')




class Asgcn(nn.Module):
    def __init__(self,configs,Tp,A):
        '''
        (B,N,C,Tr):最后应该是（B,N,Cr,Tr）->Y_hat(B,N,Tp)
        在不清楚模型的输入输出的情况下，写模型着实困难
        :param configs:dict of configs(this is also a list of dict),{'h':[config,config,],'d':[config,config],]
        :param Wh_shape:(Cr,N,Tp)
        '''
        super(Asgcn,self).__init__()
        self.module_list=nn.ModuleList([ASGCN_module(config,A) for config in configs.values()])
        #CN
        N=configs['h'][1]['num_of_nodes']
        original_channels=configs['h'][0]['in_channels']
        h_in_channels=configs['h'][-1]['num_of_time_channels']
        d_in_channels=configs['d'][-1]['num_of_time_channels']
        w_in_channles=configs['w'][-1]['num_of_time_channels']
        #使用一维卷积代替了
        #self.linear_h=nn.Linear(in_features=h_in_channels,out_features=original_channels)
        #self.Linear_d=nn.Linear(in_features=d_in_channels,out_features=original_channels)
        #self.Linear_w=nn.Linear(in_features=w_in_channles,out_features=original_channels)
        Wh_shape=(N,Tp)
        Wd_shape=(N,Tp)
        Ww_shape=(N,Tp)

# 用来模型fusion，CN
        self.Wh=nn.Parameter(torch.empty((Wh_shape)))
        torch.nn.init.xavier_uniform_(self.Wh)
        self.Wd=nn.Parameter(torch.empty((Wd_shape)))
        torch.nn.init.xavier_uniform_(self.Wd)
        self.Ww=nn.Parameter(torch.empty((Ww_shape)))
        torch.nn.init.xavier_uniform_(self.Ww)



    def forward(self,data):
        '''

        :param data: tuple or list of (b,N,Tr-1),(Xh,Xd,Xw),Xh:(b,Tr,N,C))直接产生的数据（dataloader）
        :return output:tensor,(B,Tp,N,original_channels),或者其他排布的形式(毕竟我也不太确定，还没有从头捋一遍)
        '''
        #(b,Tr,N,C)->(b,N,C,Tr)
        Yh_hat,Yd_hat,Yw_hat=[model(data[i].permute(dims=(0,2,3,1))) for i,model in enumerate(self.module_list)]#->(b,N,Cr,Tp)->(b,original_channels,N,Tp)
        #(b,N,C,Tp)->(b,N,Tp,C)->(b,N,Tp,original_channels)
        #Yh_hat=self.linear_h(Yh_hat.permute(dims=(0,1,3,2)))
        #Yd_hat=self.Linear_d(Yd_hat.permute(dims=(0,1,3,2)))
        #Yw_hat=self.Linear_w(Yw_hat.permute(dims=(0,1,3,2)))
        output=self.Wh.unsqueeze(dim=0)*Yh_hat+self.Wd.unsqueeze(dim=0)*Yd_hat+self.Ww.unsqueeze(dim=0)*Yw_hat

        return output.permute(dims=(0,2,1))   #(b,Tp,N)



# X=torch.rand(size=(3,12,4,5))   #(b,Tr,N,C)
# K=4
# #(K,C_r_1,Cr)
# theta=torch.rand(size=(K,5,7))
# S=torch.rand(size=(3,4,4))
# A=torch.rand(size=(4,4))
# #test STblock
# config1={'num_of_nodes':4,'in_channels':5,'out_channels':7,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
#         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':8,'time_step':12
#         }
# #(3,4,8,6)->b,N,C,T
# config2={'num_of_nodes':4,'in_channels':8,'out_channels':9,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
#         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':10,'time_step':12
#         }
# configs_=[config1,config2]
#
# #模型测试,只使用一个'Xh'
# configs={
#     'h':configs_,
#     'd':configs_,
#     'w':configs_
# }
# Z=(X,X,X)
#
# asgcn=Asgcn(configs,Tp=12)
# output=asgcn(Z)
# #ths shape of output : (3,12,4,5)
# print(f'the shape of tht output : {output.shape}')













# A=torch.ones((307,307))
# configs=get_configs(A,307,Th=24,Td=24,Tw=12)
# from model import Asgcn
#
# model=Asgcn(configs,12)







































