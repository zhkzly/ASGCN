# user zhengkelong
# 用作模型的参数配置
# config1={'num_of_nodes':4,'in_channels':5,'out_channels':7,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
#         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':8,'time_step':12
#         }
A=0
K=3



config_h_1={'num_of_nodes':4,'in_channels':5,'out_channels':7,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':8,'time_step':12
         }
config_h_2={'num_of_nodes':4,'in_channels':5,'out_channels':7,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':8,'time_step':12
         }

config_d_1={'num_of_nodes':4,'in_channels':5,'out_channels':7,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':8,'time_step':12
         }
config_d_2={'num_of_nodes':4,'in_channels':5,'out_channels':7,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':8,'time_step':12
         }

config_w_1={'num_of_nodes':4,'in_channels':5,'out_channels':7,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':8,'time_step':12
         }
config_w_2={'num_of_nodes':4,'in_channels':5,'out_channels':7,'gcn_conv_padding':(0,0),'gcn_conv_kernel_size':(1,1),
         'A':A,'K':K,'gcn_conv_stride':(1,1),'num_of_time_channels':8,'time_step':12
         }

config_h = [config_h_1,config_h_2]
config_d = [config_d_1,config_d_2]
config_w = [config_w_1,config_w_2]



configs = {
    'h': config_h,
    'd': config_d,
    'w': config_w
}
