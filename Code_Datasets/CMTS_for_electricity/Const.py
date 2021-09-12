import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset_name = 'electricity'
categories = {'train', 'dev', 'test'}
node_list=[1,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22
    ,23,24,25,26,27,28,30,31,32,33]
node_count = 321 

# training data rario = 1-test_ratio-dev_ratio
test_ratio = 0.1 
dev_ratio = 0.2

# data pre-process
input_time_window = 24*14
target_time_window = 24 #prediction horizon
seq_sample_step = 6
robustSTL_season_len = 24

# encoder
tcn_layer = 3
expected_time_cell_num = 29  #pruning length
tcn_hidden_channels=6

# gating function k
tanh_kernal=3

# seq2seq
lstm_hidden_size=64
lstm_input_size=4
teacher_forcing_ratio=0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')