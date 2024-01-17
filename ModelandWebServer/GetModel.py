
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from GetAttention import DL_AttentionNetwork, ML_AttentionNetwork, PositionalEncoding
import sys

src_vocab = {'P': 0, 'A': 1, 'G': 2, 'C': 3, 'U': 4}
wordsize = 5
embedingsize = 4



class TextCNN(nn.Module):
    def __init__(self, lens):
        super(TextCNN, self).__init__()

        self.encode_layer = nn.Embedding(wordsize, embedingsize)

        self.max_kernel = 6
        self.dl_cnn_out = 32
        self.hidden_dim = 16
        self.dl_att_out = 32
        self.num_layers = 1
        self.dl_attention_dim = 32
        self.dropout = 0.1


        self.D_conv = nn.Sequential(
            nn.Conv1d(in_channels=embedingsize, out_channels=2*self.dl_cnn_out, kernel_size=9,),
            nn.Conv1d(in_channels=2*self.dl_cnn_out, out_channels=self.dl_cnn_out,kernel_size=9),
            nn.BatchNorm1d(num_features=self.dl_cnn_out, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.max_kernel),
        )

        self.D_conv2 = nn.Sequential(
            nn.Conv1d(in_channels=embedingsize, out_channels=2*self.dl_cnn_out, kernel_size=20),
            nn.Conv1d(in_channels=2*self.dl_cnn_out, out_channels=self.dl_cnn_out, kernel_size=20),
            nn.BatchNorm1d(num_features=self.dl_cnn_out, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.max_kernel),
        )

        self.D_conv3 = nn.Sequential(
            nn.Conv1d(in_channels=embedingsize, out_channels=2*self.dl_cnn_out, kernel_size=49),
            nn.Conv1d(in_channels=2*self.dl_cnn_out, out_channels=self.dl_cnn_out, kernel_size=49),
            nn.BatchNorm1d(num_features=self.dl_cnn_out, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.max_kernel),
        )

        self.M_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3,),
            nn.Conv1d(in_channels=64, out_channels=32,kernel_size=3),
            nn.BatchNorm1d(num_features=32, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
        )

        self.pos = PositionalEncoding(d_model=embedingsize,max_len=lens*2)


        self.bilstm_layer1 = nn.LSTM(self.dl_cnn_out, self.hidden_dim, self.num_layers, bidirectional=True, batch_first=True)
        self.bilstm_layer2 = nn.LSTM(self.dl_cnn_out, self.hidden_dim, self.num_layers, bidirectional=True, batch_first=True)
        self.bilstm_layer3 = nn.LSTM(self.dl_cnn_out, self.hidden_dim, self.num_layers, bidirectional=True, batch_first=True)

        self.dl_attention_layer1 = DL_AttentionNetwork(self.dl_att_out, self.dl_attention_dim)
        self.dl_attention_layer2 = DL_AttentionNetwork(self.dl_att_out, self.dl_attention_dim)
        self.dl_attention_layer3 = DL_AttentionNetwork(self.dl_att_out, self.dl_attention_dim)

        self.ml_attention_layer = ML_AttentionNetwork(self.dl_att_out, self.dl_attention_dim)

        self.fc_layer = nn.Linear(4*self.dl_att_out, 2)
        self.dropout = nn.Dropout(p=self.dropout)



    def forward(self, X, ML):

        # #1.机器学习部分特征处理
        ml_input = ML.unsqueeze(1)  #[3,1,256]
        ml_conved = self.M_conv(ml_input)   #[3,32,252]
        ml_conved = ml_conved.permute(0,2,1)    #[3,252,32]

        ml_conved_len = ml_conved.shape[1]
        ml_lengths = [ml_conved_len for seq in ML]
        ml_score, ml_out = self.ml_attention_layer(ml_conved.permute(1, 0, 2), ml_lengths)  #ml_out [3,32]

        #2.深度学习部分处理特征
        x_encode = self.encode_layer(X)  # [3,8000,128]
        x_input = x_encode.permute(1, 0, 2)  # [3,128,8000]

        x_input = self.pos(x_input)  # [6000,3,4]
        x_input = x_input.permute(1, 2, 0)  # [3,4,6000]

        conved = self.D_conv(x_input)   #[3,32,998]
        conved = conved.permute(0,2,1)  #[3,998,32]
        conved_len = conved.shape[1]    #998
        lengths = []
        for seq in X:
            leng = int(len(torch.nonzero(seq))/self.max_kernel)
            if leng <= conved_len : lengths.append(leng)
            else: lengths.append(conved_len)
        x_packed_input = pack_padded_sequence(input=conved, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm_layer1(x_packed_input)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=conved_len, padding_value=0.0)
        atten_scores, atten_out = self.dl_attention_layer1(outputs.permute(1, 0, 2), lengths)


        conved2 = self.D_conv2(x_input)
        conved2 = conved2.permute(0,2,1)
        conved_len2 = conved2.shape[1]
        lengths2 = []
        for seq in X:
            leng = int(len(torch.nonzero(seq))/self.max_kernel)
            if leng <= conved_len2 : lengths2.append(leng)
            else: lengths2.append(conved_len2)
        x_packed_input2 = pack_padded_sequence(input=conved2, lengths=lengths2, batch_first=True, enforce_sorted=False)
        packed_out2, _ = self.bilstm_layer2(x_packed_input2)
        outputs2, _ = pad_packed_sequence(packed_out2, batch_first=True, total_length=conved_len2, padding_value=0.0)
        atten_scores2, atten_out2 = self.dl_attention_layer2(outputs2.permute(1, 0, 2), lengths2)


        conved3 = self.D_conv3(x_input)
        conved3 = conved3.permute(0,2,1)
        conved_len3 = conved3.shape[1]
        lengths3 = []
        for seq in X:
            leng = int(len(torch.nonzero(seq))/self.max_kernel)
            if leng <= conved_len3: lengths3.append(leng)
            else: lengths3.append(conved_len3)
        x_packed_input3 = pack_padded_sequence(input=conved3, lengths=lengths3, batch_first=True, enforce_sorted=False)
        packed_out3, _ = self.bilstm_layer3(x_packed_input3)
        outputs3, _ = pad_packed_sequence(packed_out3, batch_first=True, total_length=conved_len3, padding_value=0.0)
        atten_scores3, atten_out3 = self.dl_attention_layer3(outputs3.permute(1, 0, 2), lengths3)

        #深度学习处理后的atten值
        atten_dl = torch.cat((atten_out,atten_out2,atten_out3,ml_out),1)
        #atten_dl = torch.cat((atten_out, atten_out2, atten_out3), 1)


        output = self.fc_layer(atten_dl)
        output = self.dropout(output)

        return output
