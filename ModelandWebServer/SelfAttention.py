import os
import random
import numpy as np
import torch.nn as nn
import torch
device = torch.device('cpu')


# Padding Should be Zero
src_vocab = {'P': 0, 'A': 1, 'G': 2, 'C': 3, 'U': 4}
src_vocab_size = len(src_vocab)

# Transformer Parameters
heads = 5
d_model = 32
num_classes = 2
seed = 42


def set_seed(seed): # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)


#获取序列中的填充位置
def get_attn_pad_mask(seq_q, heads, lenghts):

    batch_size, len_q,_ = seq_q.size()
    seq = np.zeros((batch_size,len_q))


    for i in range(len(lenghts)):
        for j in range(lenghts[i]):
            seq[i][j] = seq[i][j] + 1

    seq = torch.tensor(seq).cuda()

    # eq(zero) is PAD token
    #pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    pad_attn_mask = seq.unsqueeze(1)

    return pad_attn_mask.expand(batch_size, heads, len_q)  # [batch_size, len_q, len_k]


#进行DotProduct
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.tanh = torch.tanh


    def forward(self, Q, K):

        scores = torch.matmul(Q, K.transpose(-1, -2))   # scores : [batch_size, n_heads, len_q, len_k]

        scores = self.tanh(scores)

        return scores


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)

    def forward(self, input_Q, input_K):

        Q = self.W_Q(input_Q)
        K = self.W_K(input_K)

        attn = ScaledDotProductAttention()(Q, K)

        return attn


class Self_Attention(nn.Module):
    def __init__(self):
        super(Self_Attention, self).__init__()

        self.W = nn.Parameter(torch.rand(heads, 997))

        self.enc_self_attn = MultiHeadAttention()

        self.fc = nn.Linear(heads * d_model, num_classes)

    def forward(self, enc_inputs, lenghts):
        '''
        enc_inputs: [batch_size, src_len]
        '''

        enc_self_attn = self.enc_self_attn(enc_inputs, enc_inputs)   #[12,800,800]

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, heads, lenghts)  # [batch_size, src_len, src_len] [12,5,800]  为之后进行mask做准备

        W_Atten = torch.matmul(self.W, enc_self_attn)   #[12,5,800]

        score = W_Atten.masked_fill_(enc_self_attn_mask == 0, -np.inf*1000)  # Fills elements of self tensor with value where mask is True.
        score = score/1000

        score_normal = nn.Softmax(dim=-1)(score)

        outputs = torch.matmul(score_normal, enc_inputs)

        return outputs
