

import torch
from argparse import ArgumentParser
import numpy as np
from utils import data_preprocession,Mydataset,calculate_adj_matrix,get_configs,evaluate
from torch.utils.data import DataLoader
from model import Asgcn
from train import Trainer
import matplotlib.pyplot as plt
#torch.autograd.set_detect_anomaly(True)

argp=ArgumentParser(description='training ASGCN')
argp.add_argument('data_path',help='this is used to load the original data')
argp.add_argument('mode',help='this is used to choose whether train or test',type=str)  
argp.add_argument('--save_params_to',default='params_of_model',help='this is used to decide where to save the params')
argp.add_argument('--save_tensorboard_path',default='loss',help='this is used to decide where to save the data of tensorboard')
argp.add_argument('--batch_size',default=16,help='this is used to choose the batch size of data',type=int)
argp.add_argument('--epochs',default=100,type=int)
argp.add_argument('--lr',default=0.001,type=float,help='learning rate')
argp.add_argument('--is_shuffle',default=False,help='decide whether shuffle the dataset or not')
argp.add_argument('--is_distribute',default=False,help='decide whether to use the distributed training')
argp.add_argument('--save_prediction_to',default='prediction_result')
argp.add_argument('--loss_history_path',default='loss_history_path')
argp.add_argument('--dtype',default=torch.float32)
argp.add_argument('--adja_path',default='../data/PEMS08/distance.csv',type=str)
argp.add_argument('--pretrain_model_params_path',default='params_of_model',type=str)
argp.add_argument('--Tp',default=12,type=int)
argp.add_argument('--Th',default=36,type=int)
argp.add_argument('--Td',default=12,type=int)
argp.add_argument('--Tw',default=12,type=int)

args=argp.parse_args()

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run(args):
    '''
    this is used to train and test the model
    :param args:
    :return:
    '''
    mode=args.mode
    data_path=args.data_path
    epochs=args.epochs
    is_shuffle=args.is_shuffle
    is_distribute=args.is_distribute
    lr=args.lr
    batch_size=args.batch_size
    save_tensorboard_to=args.save_tensorboard_path
    save_params_to=args.save_params_to
    save_prediction_to=args.save_prediction_to
    loss_history_path=args.loss_history_path
    adja_path=args.adja_path
    pretrain_model_params_path=args.pretrain_model_path
    dtype=args.dtype
    Tp=args.Tp
    Th=args.Th
    Tw=args.Tw
    Td=args.Td



    #following is the preprocession of the data
    #shape{time slice:T_p,nodes:N,features:F}   
   
    #features,this different from the original paper
   
    # in the paper ,the author just mean the data
    if mode=='train':
          data=data_preprocession(data_file_path=data_path,Th=Th,Td=Td,Tw=Tw,Tp=Tp)
          # has_nan=torch.isnan(data['train_norm_set']['Xw']).any()
          # print(has_nan)
          # assert False
          
          adja=np.loadtxt(open(adja_path,'rb'),delimiter=',',skiprows=1,usecols=[0,1])
          A=calculate_adj_matrix(adja)
          A=torch.from_numpy(A).type(dtype)
          #print(A.shape[0])
          #assert False
          configs=get_configs(num_of_nodes=A.shape[0],Th=Th,Td=Td,Tw=Tw,dtype=dtype)
          my_model=Asgcn(configs,Tp=Tp,A=A)
          optimizer=torch.optim.Adam(params=my_model.parameters(),lr=lr)
          loss=torch.nn.MSELoss()
          # for epochs in epochs:
          #     for batch_train_set in data_loader:
          #         print(batch_train_set[0].shape,batch_train_set[1].shape)
          #         pred=my_model(batch_train_set[:3])
          #         print('stop successfully')
          #         print(f'the shape of the model :{pred.shape}')
          #         break
          #print(f'epochs:{epochs}')
          loss_history_array=Trainer(model=my_model,data=data['train_norm_set'],optimizer=optimizer,save_params_to=save_params_to,loss=loss,epochs=epochs
                  ,batch_size=batch_size,shuffle=is_shuffle,distribute=is_distribute,device=device,loss_history_path=loss_history_path
                  )
    
      
          #plt.plot(loss_history_array[:,1])
          #plt.ylabel('loss')
          #plt.xlabel('iter')
          #plt.savefig('loss.jpg')
          #plt.show()
    elif mode=='val':
          with torch.no_grad():
              
              data = data_preprocession(data_file_path=data_path, Th=Th, Td=Td, Tw=Tw, Tp=Tp)
              # has_nan=torch.isnan(data['train_norm_set']['Xw']).any()
              # print(has_nan)
              # assert False
              test_dataset=data['val_norm_set']
              adja = np.loadtxt(open(adja_path, 'rb'), delimiter=',', skiprows=1, usecols=[0, 1])
              A = calculate_adj_matrix(adja)
              A = torch.from_numpy(A).type(dtype)
              configs = get_configs(num_of_nodes=A.shape[0], Th=Th, Td=Td, Tw=Tw, dtype=dtype)
              my_model = Asgcn(configs, Tp=Tp, A=A)
              result=evaluate(my_model,test_dataset,data['stats']['mean'],data['stats']['std'],model_params_path=save_params_to,batch_size=batch_size)
              # for epochs in epochs:
              #     for batch_train_set in data_loader:
              #         print(batch_train_set[0].shape,batch_train_set[1].shape)
              #         pred=my_model(batch_train_set[:3])
              #         print('stop successfully')
              #         print(f'the shape of the model :{pred.shape}')
              #         break
              # print(f'epochs:{epochs}')
              print(f'the final result:rmes:{result[0]},mes:{result[1]},mase:{result[2]}')
      
      
    elif mode=='test':
      
          with torch.no_grad():
              data = data_preprocession(data_file_path=data_path, Th=Th, Td=Td, Tw=Tw, Tp=Tp)
              # has_nan=torch.isnan(data['train_norm_set']['Xw']).any()
              # print(has_nan)
              # assert False
              test_dataset=data['test_norm_set']
              adja = np.loadtxt(open(adja_path, 'rb'), delimiter=',', skiprows=1, usecols=[0, 1])
              A = calculate_adj_matrix(adja)
              A = torch.from_numpy(A).type(dtype)
              configs = get_configs(num_of_nodes=A.shape[0], Th=Th, Td=Td, Tw=Tw, dtype=dtype)
              my_model = Asgcn(configs, Tp=Tp, A=A)
              result=evaluate(my_model,test_dataset,data['stats']['mean'],data['stats']['std'],model_params_path=save_params_to,batch_size=batch_size)
              # for epochs in epochs:
              #     for batch_train_set in data_loader:
              #         print(batch_train_set[0].shape,batch_train_set[1].shape)
              #         pred=my_model(batch_train_set[:3])
              #         print('stop successfully')
              #         print(f'the shape of the model :{pred.shape}')
              #         break
              # print(f'epochs:{epochs}')
              print(f'the final result:rmes:{result[0]},mes:{result[1]},mase:{result[2]}')
      
      
      
    elif mode=='finetune':
        
          data=data_preprocession(data_file_path=data_path,Th=Th,Td=Td,Tw=Tw,Tp=Tp)
          # has_nan=torch.isnan(data['train_norm_set']['Xw']).any()
          # print(has_nan)
          # assert False
          
          adja=np.loadtxt(open(adja_path,'rb'),delimiter=',',skiprows=1,usecols=[0,1])
          A=calculate_adj_matrix(adja)
          A=torch.from_numpy(A).type(dtype)
          #print(A.shape[0])
          #assert False
          configs=get_configs(num_of_nodes=A.shape[0],Th=Th,Td=Td,Tw=Tw,dtype=dtype)
          my_model=Asgcn(configs,Tp=Tp,A=A)
          my_model.load_state_dict(torch.load(pretrain_model_params_path))
          optimizer=torch.optim.Adam(params=my_model.parameters(),lr=lr)
          loss=torch.nn.MSELoss()
          # for epochs in epochs:
          #     for batch_train_set in data_loader:
          #         print(batch_train_set[0].shape,batch_train_set[1].shape)
          #         pred=my_model(batch_train_set[:3])
          #         print('stop successfully')
          #         print(f'the shape of the model :{pred.shape}')
          #         break
          #print(f'epochs:{epochs}')
          loss_history_array=Trainer(model=my_model,mode='finetune',data=data['train_norm_set'],optimizer=optimizer,save_params_to=save_params_to,loss=loss,epochs=epochs
                  ,batch_size=batch_size,shuffle=is_shuffle,distribute=is_distribute,device=device,loss_history_path=loss_history_path,model_params_path=model_params_path
                  )
      
    else:
          raise ValueError('wrong input value!')          

if __name__=='__main__':
    run(args)







