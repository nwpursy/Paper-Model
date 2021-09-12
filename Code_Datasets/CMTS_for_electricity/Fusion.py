# encoding: utf-8
import torch
from CMTS_for_electricity.Sublayers import my_tanh
from CMTS_for_electricity import Const
import torch.nn as nn
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# torch.set_printoptions(profile="full", sci_mode=False)

class FusionLayer(object):
    def __init__(self):
        pass

    @staticmethod
    def fusion(input_tensor, index):
        node_size = input_tensor.size()[1] 
        with torch.no_grad():
            x = input_tensor[:, index, :] 
            for j in range(node_size):
                y = input_tensor[:, j, :]
                if j == 0:
                    distance = torch.cosine_similarity(x, y, dim=1).view(1, -1)
                else:
                    distance = torch.cat([distance, torch.cosine_similarity(x, y, dim=1).view(1, -1)], dim=0)
            interactions = FusionLayer.gated_mechanism(distance)
        interactions = interactions.transpose(1, 0)
        interactions2 = interactions.view(interactions.size(0), 1, interactions.size(1))
        fusion_tensor = torch.bmm(interactions2, input_tensor)/Const.node_count
        return fusion_tensor

    @staticmethod
    def gated_mechanism(inputs):
        gate = my_tanh(k=Const.tanh_kernal)
        output_data = gate(inputs)
        coupled_structure = inputs.mul(output_data)
        return coupled_structure

    @staticmethod
    def coupling(inputs, index):
        inputs = inputs.view(inputs.size()[0], inputs.size()[1], -1, inputs.size()[-1])
        feature_size = inputs.size()[2]
        for i in range(feature_size):
            fusion_tensor = FusionLayer.fusion(inputs[:, :, i, :], index).squeeze(1)
            if i == 0:
                context = fusion_tensor.clone().detach()
            else:
                context = torch.cat([context, fusion_tensor.clone().detach()], dim=1)
        return context


if __name__ == '__main__':
    a = torch.randn(300, 40, 2, 3, 10)
    index_column = 0
    context=FusionLayer.coupling(a, index_column)
    print(context.size())