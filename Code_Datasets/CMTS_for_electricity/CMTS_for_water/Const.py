import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset_name = 'water_pressure'
categories = {'train', 'dev', 'test'}
node_count = 30  

test_ratio = 0.1
dev_ratio = 0.2

# data pre-process
input_time_window = 24*14
target_time_window = 24
seq_sample_step = 6
robustSTL_season_len = 24

# encoder
tcn_layer = 3
expected_time_cell_num = 29
tcn_hidden_channels=6
tanh_kernal=3

# seq2seq
lstm_hidden_size=64
lstm_input_size=4
teacher_forcing_ratio=0.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')