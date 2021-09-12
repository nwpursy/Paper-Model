import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

from CMTS_for_water import Const
from CMTS_for_water import Utils

mpl.rcParams['font.sans-serif'] = ['SimHei']


def csv2DataFrame(fileName, colNum, parentPath='../resource/data/'):
    return pd.read_csv(parentPath + fileName, names=range(colNum))
    # return pd.read_csv(parentPath + fileName, names=range(colNum),usecols=Const.node_list)

class DataAccess:

    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        self.rowNum = dataFrame.shape[0]
        self.columnNum = dataFrame.shape[1]
        # self.test_size=int(self.rowNum*Const.test_ratio)
        # self.non_test_size=self.rowNum-self.test_size

    def createInoutSequences(self, input_time_window, target_time_window, seq_sample_step):
        input_seq = []
        target_seq = []
        L = self.rowNum
        T = input_time_window
        I = target_time_window
        for i in range(T, L - I + 1, seq_sample_step):
            inp = torch.FloatTensor(self.dataFrame[i - T:i].values).transpose(0, 1).to(Const.device)
            input_seq.append(inp)
            target = torch.FloatTensor(self.dataFrame[i:i + I].values).transpose(0, 1).to(Const.device)
            target_seq.append(target)
        self.input_seq = torch.stack(input_seq)
        self.target_seq = torch.stack(target_seq)

    def divideInoutSequences(self, dev_ratio, test_ratio):
        # shuffle in all sequences
        sample_size = self.input_seq.size(0)
        seed = np.arange(sample_size)
        np.random.shuffle(seed)
        self.input_seq = self.input_seq[seed, :, :]
        self.target_seq = self.target_seq[seed, :, :]

        # divide
        dev_size = int(sample_size * dev_ratio)
        test_size = int(sample_size * test_ratio)
        train_size = sample_size - dev_size - test_size

        self.train_input_seq = self.input_seq[:train_size, :, :]
        self.train_target_seq = self.target_seq[:train_size, :, :]

        self.dev_input_seq = self.input_seq[train_size:train_size + dev_size, :, :]
        self.dev_target_seq = self.target_seq[train_size:train_size + dev_size, :, :]

        self.test_input_seq = self.input_seq[-test_size:, :, :]
        self.test_target_seq = self.target_seq[-test_size:, :, :]

        self.train_size = train_size
        self.dev_size = dev_size
        self.test_size = test_size

    def shuffle(self, option):
        filtered_seq = self.__getattribute__('{}_filtered_seq'.format(option))
        target_seq = self.__getattribute__('{}_target_seq'.format(option))
        sample_size = filtered_seq.size(0)
        seed = np.arange(sample_size)
        np.random.shuffle(seed)
        self.__setattr__('{}_filtered_seq'.format(option), filtered_seq[seed])
        self.__setattr__('{}_target_seq'.format(option), target_seq[seed])

    def businessProcess(self, isFirstRun=False, column='all'):
        if isFirstRun:
            self.createInoutSequences(Const.input_time_window, Const.target_time_window, Const.seq_sample_step)
            self.divideInoutSequences(Const.dev_ratio, Const.test_ratio)
            self.preprocess()
            pass
        else:
            for category in Const.categories:
                self.loadFilteredData(category, dataset_name=Const.dataset_name, device=Const.device,column=column)

    def plotSomeColumns(self, columnNames):
        x_data = self.dataFrame.index
        fig, ax = plt.subplots()
        for col in columnNames:
            y_data = self.dataFrame[col]
            ax.plot(x_data, y_data)
        plt.grid()
        plt.legend()
        plt.show()

    def preprocess(self):
        for category in Const.categories:
            self.filtering(category, Const.robustSTL_season_len)
            self.saveFilteredData(category, Const.dataset_name)
        pass

    def filtering(self, category, season_len):
        input_seq = self.__getattribute__('{}_input_seq'.format(category))
        size = input_seq.size(0)
        filtered_seq = []

        print(category, '开始')
        for s_i in tqdm(range(size)):
            inp = input_seq[s_i]
            filteredInp = []
            for v_i in range(self.columnNum):
                # result = rstl.RobustSTL(inp[v_i].numpy(), season_len)
                rd=sm.tsa.seasonal_decompose(inp[v_i].numpy(),period=season_len
                                             ,extrapolate_trend='freq')
                result = [rd.observed, rd.trend, rd.seasonal, rd.resid]
                filteredInp.append(result)
                pass
            filtered_seq.append(filteredInp)
        filtered_seq = torch.Tensor(filtered_seq)
        self.__setattr__('{}_filtered_seq'.format(category), filtered_seq)

    def saveFilteredData(self, category, dataset_name, parentPath='../resource/preprocess/', suffix='.pt'):
        filtered_seq = self.__getattribute__('{}_filtered_seq'.format(category))
        target_seq = self.__getattribute__('{}_target_seq'.format(category))
        torch.save([filtered_seq, target_seq], '{}{}${}h_{}{}'.format(
            parentPath, dataset_name,Const.target_time_window, category, suffix))

    def loadFilteredData(self, category, dataset_name, device, column='all', parentPath='../resource/preprocess/', suffix='.pt'):
        [filtered_seq, target_seq] = torch.load('{}{}${}h_{}{}'.format(
            parentPath, dataset_name,Const.target_time_window, category, suffix),map_location=device)

        self.__setattr__('{}_filtered_seq'.format(category), filtered_seq)
        if column=='all':
            self.__setattr__('{}_target_seq'.format(category), target_seq)
        else:
            self.__setattr__('{}_target_seq'.format(category), target_seq[:,column])

    def divideIntoMiniBatch(self, nameList, batch_size):
        result = []
        for name in nameList:
            batch = Utils.divideDataIntoMiniBatch(self.__getattribute__(name), batch_size)
            result.append(batch)
        return result


if __name__ == '__main__':
    df = csv2DataFrame(Const.dataset_name+'.csv', Const.node_count)
    da = DataAccess(df)

    da.businessProcess(isFirstRun=True)

    print(da.train_filtered_seq.size())
    print(da.train_target_seq.size())
    print(da.dev_filtered_seq.size())
    print(da.dev_target_seq.size())
    print(da.test_filtered_seq.size())
    print(da.test_target_seq.size())
