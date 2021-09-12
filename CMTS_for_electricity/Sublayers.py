import torch
import torch.nn as nn
from CMTS_for_electricity import Const
from CMTS_for_electricity.Utils import calculate_laplacian_with_self_loop
import argparse
from torch.functional import F
import matplotlib.pyplot as plt


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        results = super(CausalConv1d, self).forward(inputs)
        return results


class Inception_Temporal_Layer(nn.Module):
    def __init__(self, num_stations, In_channels=1, Hid_channels=1):
        super(Inception_Temporal_Layer, self).__init__()
        self.temporal_conv1 = CausalConv1d(In_channels, Hid_channels, 6, dilation=1, stride=3)
        self.temporal_conv2 = CausalConv1d(Hid_channels, Hid_channels, 4, dilation=2, stride=1)
        self.temporal_conv3 = CausalConv1d(Hid_channels, Hid_channels, 13, dilation=4, stride=2)
        self.num_stations = num_stations
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        output = []
        filter_out_size = inputs.size(2)
        for s_i in range(self.num_stations):
            output_i = []
            for f_j in range(filter_out_size):
                inputs_ij = inputs[:, s_i, f_j].transpose(1, 2)
                output_1_ij = self.act(self.temporal_conv1(inputs_ij))
                output_2_ij = self.act(self.temporal_conv2(output_1_ij))
                output_3_ij = self.act(self.temporal_conv3(output_2_ij))
                max_time_cell_num = output_3_ij.size(-1)
                assert (max_time_cell_num >= Const.expected_time_cell_num)
                output_ij = torch.stack([output_1_ij[:, :, :Const.expected_time_cell_num].transpose(1, 2)
                                            , output_2_ij[:, :, :Const.expected_time_cell_num].transpose(1, 2)
                                            , output_3_ij[:, :, :Const.expected_time_cell_num].transpose(1, 2)], dim=1)
                output_i.append(output_ij)
            output_i = torch.stack(output_i, dim=1)
            output.append(output_i)
        output = torch.stack(output, dim=1)
        return output



class my_tanh(nn.Module):
    def __init__(self, k=1):
        super(my_tanh, self).__init__()
        self.k = k

    def forward(self, x):
        x = torch.tanh(self.k * x)
        x = x * x
        return x



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),  # b, 16(高度), 26, 26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # b, 32, 24, 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 32, 12, 12
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # b, 64, 10, 10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # b, 128, 8, 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 128, 4, 4
        )

        self.out = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),  # (input, output)
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),  # (input, output)
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)  # (input, output)
        )

    def forward(self, x):
        x = self.layer1(x)  # (batch, 16, 26, 26) -> (batchsize)
        x = self.layer2(x)  # (batch, 32, 12, 12)
        x = self.layer3(x)  # (batch, 64, 10, 10)
        x = self.layer4(x)  # (batch, 128, 4, 4)
        x = x.view(x.size(0), -1) 
        x = self.out(x)
        return x


class GCN(nn.Module):
    def __init__(self, adj, input_dim: int, output_dim: int, **kwargs):
        super(GCN, self).__init__()
        self.register_buffer('laplacian', calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim  # seq_len for prediction
        self._output_dim = output_dim  # hidden_dim for prediction
        self.weights = nn.Parameter(torch.FloatTensor(self._input_dim, self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('tanh'))

    def forward(self, inputs):
        # (batch_size, seq_len, num_nodes)
        batch_size = inputs.shape[0]
        # (num_nodes, batch_size, seq_len)
        inputs = inputs.transpose(0, 2).transpose(1, 2)
        # (num_nodes, batch_size * seq_len)
        inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))
        # AX (num_nodes, batch_size * seq_len)
        ax = self.laplacian @ inputs
        # (num_nodes, batch_size, seq_len)
        ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
        # (num_nodes * batch_size, seq_len)
        ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
        # act(AXW) (num_nodes * batch_size, output_dim)
        outputs = torch.tanh(ax @ self.weights)
        # (num_nodes, batch_size, output_dim)
        outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
        # (batch_size, num_nodes, output_dim)
        outputs = outputs.transpose(0, 1)
        return outputs

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {
            'num_nodes': self._num_nodes,
            'input_dim': self._input_dim,
            'output_dim': self._output_dim
        }



class MLP(torch.nn.Module): 
    def __init__(self):
        super(MLP, self).__init__()  #
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 128) 
        self.fc3 = torch.nn.Linear(128, 10) 

    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        dout = F.sigmoid(self.fc3(dout), dim=1)
        return dout


class Attention(nn.Module):
    def __init__(self,enc_vec_size,dec_vec_size,attn_hidden_size):
        super(Attention,self).__init__()
        self.attn_inner=nn.Linear(enc_vec_size+dec_vec_size,attn_hidden_size)
        self.attn_outer=nn.Linear(attn_hidden_size,1,bias=False)
    def forward(self,elements,compare):
        enc_element_num=elements.size(0)
        ss=compare.repeat(enc_element_num,1,1)
        sh=torch.cat([ss,elements],dim=2) # shape=(enc_element_num,batch,enc_vec_size+dec_vec_size)
        attn_middle = torch.tanh(self.attn_inner(sh))
        attn_weight = F.softmax(self.attn_outer(attn_middle), dim=0)
        attn_weight = attn_weight.transpose(0, 1).transpose(1, 2)
        context = torch.bmm(attn_weight, elements.transpose(0, 1))
        attn_weight = attn_weight.view(-1, enc_element_num) # shape=(batch,enc_element_num)
        context=context.squeeze(dim=1)
        return context,attn_weight
        pass

if __name__ == '__main__':
    a=torch.randn(300,8,4,150,1)
    tcn=Inception_Temporal_Layer(8,Hid_channels=7)
    out=tcn(a)
    print(out.size())
