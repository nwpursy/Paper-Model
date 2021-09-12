import torch
import torch.nn as nn
import random

from CMTS_for_electricity import Const


class EncoderLSTM(nn.Module):
    def __init__(self,hidden_size, input_size):
        super(EncoderLSTM,self).__init__()
        self.hidden_size=hidden_size
        self.input_size=input_size;
        self.lstm=nn.LSTM(input_size,hidden_size,batch_first=True)
    def forward(self,x):
        batch_size=x.size(0)
        hidden_cell=self.initHiddenCell(batch_size)
        output,hidden_cell=self.lstm(x,hidden_cell)
        h=hidden_cell[0]
        c=hidden_cell[1]
        h=h[0]
        c=c[0]
        hidden_cell=(h,c)
        # print('enc_output size:',output.size())
        return output, hidden_cell
    def initHiddenCell(self,batch_size):
        return (torch.zeros(1,batch_size,self.hidden_size,device=Const.device),
                            torch.zeros(1,batch_size,self.hidden_size,device=Const.device))


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size=1):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm=nn.LSTMCell(output_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, y, hidden_cell, attn_context=None):
        yc=y
        if attn_context!=None:
            yc = torch.cat((attn_context, y), dim=1)

        hidden_cell = self.lstm(yc, hidden_cell)
        y_hat=self.out(hidden_cell[0]).view(-1,self.output_size)
        return y_hat, hidden_cell

class UniVariateSeq2Seq(nn.Module):
    def __init__(self,context_size=0):
        super(UniVariateSeq2Seq, self).__init__()
        self.encoder=EncoderLSTM(Const.lstm_hidden_size,Const.lstm_input_size)
        self.decoder=DecoderLSTM(Const.lstm_hidden_size+context_size)
        if context_size>0:
            self.context_size=context_size
            self.context_combine=nn.Linear(Const.lstm_hidden_size,Const.lstm_hidden_size+context_size)

    def forward(self, input_tensor,target_tensor,context=None,teacher_forcing_ratio=Const.teacher_forcing_ratio):
        assert len(input_tensor)==len(target_tensor)
        batch_size=len(input_tensor)
        dec_seq_length = target_tensor.size(1)
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)

        output_size=target_tensor.size(2)
        dec_input=input_tensor[:,-1,0:1] 

        decoder_hidden=self.context_combination(encoder_hidden,context)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        dec_output=[]
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(dec_seq_length):
                (y_hat, decoder_hidden) = self.decoder(dec_input, decoder_hidden)
                dec_output.append(y_hat)
                dec_input = target_tensor[:, di].view(batch_size, output_size)
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(dec_seq_length):
                (y_hat, decoder_hidden) = self.decoder(dec_input, decoder_hidden)
                dec_output.append(y_hat)
                dec_input = y_hat.detach()
        dec_output=torch.stack(dec_output,dim=1)
        return dec_output

    def context_combination(self,encoder_hidden,context=None):
        if self.context_size>0 and context!=None:
            (h, c) = (encoder_hidden[0], encoder_hidden[1])
            assert h.is_same_size(c)
            assert h.size(0)==context.size(0)
            dec_h0=torch.cat([h,context],dim=1)
            dec_c0=self.context_combine(c)
            return (dec_h0,dec_c0)
        else:
            return encoder_hidden

    def evaluate(self,input_tensor,prediction_length,context=None):
        self.eval()
        batch_size = len(input_tensor)
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            # dec_input = torch.zeros(batch_size, 1, device=Const.device)
            dec_input = input_tensor[:, -1, 0:1]
            decoder_hidden = self.context_combination(encoder_hidden, context)
            dec_output = []
            for di in range(prediction_length):
                (y_hat, decoder_hidden) = self.decoder(dec_input, decoder_hidden)
                dec_output.append(y_hat)
                dec_input = y_hat.detach()
            dec_output = torch.stack(dec_output, dim=1)
            return dec_output

if __name__=='__main__':
    seq2seq=UniVariateSeq2Seq(context_size=60)
    x=torch.randn(309,336,4)
    y=torch.randn(309,12,1)
    output=seq2seq(x,y,context=torch.randn(309,60))
    print(output.size())