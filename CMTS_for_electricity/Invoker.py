import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import os
import random
import numpy as np
import pandas as pd

from CMTS_for_electricity import DataPreparation as dp
from CMTS_for_electricity import Const
from CMTS_for_electricity.Model import Seq2SeqBasedModel
from CMTS_for_electricity.Utils import model_size
from CMTS_for_electricity.Train import train, test


def configuration(target_time_window,column='all'):
    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder_in_channel", type=int, default=1)
    parser.add_argument("-encoder_out_channel", type=int, default=1)
    parser.add_argument("-epoch", type=int, default=80)
    parser.add_argument("-batch_size", type=int, default=100)
    parser.add_argument("-log", default=None)
    parser.add_argument("-save_model", default=None)
    parser.add_argument("-save_mode", type=str, choices=["all", "best"], default="best")
    option = parser.parse_args()
    option.save_model = "../resource/Result_Models/CMTS4elec/complete/{}h/CMTS@{}".format(target_time_window,column)
    return option

def main(da: dp.DataAccess,target_time_window,column='all',case=0,model_file_name=None):
    option = configuration(target_time_window,column)

    model=Seq2SeqBasedModel(column,option.encoder_in_channel,option.encoder_out_channel).to(Const.device)
    if model_file_name!=None:
        print("Loading Saved Model")
        checkpoint = torch.load(model_file_name)
        model_state_dict = checkpoint["model"]
        model = model.to(Const.device)
        model.load_state_dict(model_state_dict)

    model_size(model)
    # cudnn.benchmark = True

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, [15,30,50], gamma=0.316)
    option.log = option.save_model
    train(model, da, criterion, optimizer, scheduler, option, Const.device)
    mae,rmse=test_main(option.save_model + ".pkl", da, column, case=case)
    return mae,rmse

def test_main(model_file_name, da: dp.DataAccess,column,case=0):
    print("Loading Saved Model")
    checkpoint = torch.load(model_file_name)

    model_state_dict = checkpoint["model"]
    option = checkpoint["setting"]

    model_structure=Seq2SeqBasedModel(column,option.encoder_in_channel,option.encoder_out_channel).to(Const.device)
    model_structure = model_structure.to(Const.device)
    model_structure.load_state_dict(model_state_dict)

    print("Loading Testing Dataset")
    criterion = nn.MSELoss()
    mae, rmse, loss = test(model_structure, da, criterion, Const.device,case=case)
    print("[ Results ]\n  - loss:{:2.8f}  - mae:{:2.8f}  - rmse:{:2.8f}".format(loss, mae, rmse))
    return mae,rmse


def seed_torch(seed=102):
    print('set seed')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    log_mae='../resource/Result_Models/CMTS4elec/complete/{}h/mae.log'.format(Const.target_time_window)
    log_rmse='../resource/Result_Models/CMTS4elec/complete/{}h/rmse.log'.format(Const.target_time_window)

    # column=7 # index of node
    # print('node', column)
    # df = pd.DataFrame()
    # da = dp.DataAccess(df)
    # da.businessProcess(isFirstRun=False, column=column)
    # test_main("../resource/Result_Models/CMTS4elec/complete/12h/CMTS@{}.pkl".format(column), da, column, case=250)  # test

    for column in range(Const.node_count):
        print('node',column)
        df=pd.DataFrame()
        da = dp.DataAccess(df)
        da.businessProcess(isFirstRun=False,column=column)
        seed_torch()
        # main(da,column,case=28,model_file_name="../resource/Result_Models/CMTS@{}.pkl".format(column)) #reload
        mae,rmse=main(da, Const.target_time_window, column, case=28) #restart
        # test_main("../resource/Result_Models/CMTS4elec/complete/12h/CMTS@{}.pkl".format(column), da, column, case=165) #test

        with open(log_mae, "a") as a, open(log_rmse, "a") as r:
            a.write("{}\t{:2.8f}\n".format(column, mae))
            r.write("{}\t{:2.8f}\n".format(column, rmse))