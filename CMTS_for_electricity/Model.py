import torch
import torch.nn as nn

from CMTS_for_electricity.Sublayers import Inception_Temporal_Layer
from CMTS_for_electricity.Fusion import FusionLayer
from CMTS_for_electricity import Const
from CMTS_for_electricity.UniVariateSeq2Seq import UniVariateSeq2Seq


class Seq2SeqBasedModel(nn.Module):
    def __init__(self,index_of_node,encoder_in_channel=1, encoder_out_channel=1):
        super(Seq2SeqBasedModel, self).__init__()
        self.index_of_node=index_of_node
        self.context_size=2*Const.tcn_layer*Const.expected_time_cell_num

        self.uvs2s=UniVariateSeq2Seq(self.context_size)
        self.cg=ContextGenerator(Const.node_count,encoder_in_channel,encoder_out_channel)

    def forward(self, inputs, target):
        context=self.cg(inputs,self.index_of_node)
        inputs=inputs[:,self.index_of_node].transpose(-1,-2)
        target=target.unsqueeze(-1)
        ret=self.uvs2s(inputs,target,context=context).squeeze(-1)
        return ret
        
    def evaluate(self, inputs, prediction_length):
        with torch.no_grad():
            context = self.cg(inputs, self.index_of_node)
            inputs = inputs[:, self.index_of_node].transpose(-1, -2)
            ret=self.uvs2s.evaluate(inputs,prediction_length,context=context).squeeze(-1)
            return ret


class ContextGenerator(nn.Module):
    def __init__(self, num_stations, encoder_in_channel=1, encoder_out_channel=1):
        super(ContextGenerator, self).__init__()

        self.encoder = Prediction_Encoder(num_stations, in_channels=encoder_in_channel
                                          ,Hid_channels=Const.tcn_hidden_channels
                                          ,out_channels=encoder_out_channel)
        self.decoder = Prediction_Decoder()

    def forward(self, inputs, index_of_node):
        inputs=inputs.unsqueeze(-1) # 1 in channel
        middle=self.encoder(inputs)
        out=self.decoder(middle,index_of_node)
        return out


class Prediction_Encoder(nn.Module):
    def __init__(self, num_stations, in_channels=1, Hid_channels=1, out_channels=1):
        super(Prediction_Encoder, self).__init__()
        self.tcn = Inception_Temporal_Layer(num_stations, In_channels=in_channels, Hid_channels=Hid_channels)
        self.act = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.num_stations = num_stations
        self.in_channels = in_channels

    def forward(self, inputs):
        inputs=inputs[:,:,0:2,:,:] # origin & trend
        batch_size, num_stations, filter_out_size, seq_len, in_channels = inputs.size()
        inputs=inputs[:,:,:,range(seq_len - 1, -1, -1)]

        assert num_stations == self.num_stations
        assert in_channels == self.in_channels
        temporal_feature = self.tcn(inputs)
        out=temporal_feature.sum(dim=-1)
        return out


class Prediction_Decoder(nn.Module):
    def __init__(self):
        super(Prediction_Decoder, self).__init__()

    def forward(self, inputs, index_of_node):
        context=FusionLayer.coupling(inputs,index_of_node)
        return context

if __name__ == '__main__':
    from CMTS import DataPreparation as dp
    df = dp.csv2DataFrame(Const.dataset_name + '.csv', Const.node_count)
    da = dp.DataAccess(df)
    column=13
    da.businessProcess(isFirstRun=False,column=column)
    model=Seq2SeqBasedModel(column).to(Const.device)
    result=model(da.train_filtered_seq,da.train_target_seq)
    print(result.size())
