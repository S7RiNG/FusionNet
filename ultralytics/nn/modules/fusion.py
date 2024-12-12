import torch.nn as nn
import torch
import math
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch.nn.init import constant_, xavier_uniform_

__all__ = ('FuisonBlock', "FusionConcatInput", 'FusionSequence', 'FusionSplitResult', 'FusionLinear')

class FusionSequence(nn.Module):
    def __init__(self, d_model, repeat=1, n_head=1, d_ff=None):
        super().__init__()
        self.d_model = d_model
        d_ff = d_model if d_ff is None else d_ff
        self.faltten = nn.Flatten(2, -1)     
        self.pe = PositionalEncoding1D(d_model)
        sq1, sq2 = [], []

        for _ in range(repeat):
            sq1.append(FuisonBlock(d_model, d_ff, n_head))
            sq2.append(FuisonBlock(d_model, d_ff, n_head))
        self.fusionseq1 = nn.Sequential(*sq1)
        self.fusionseq2 = nn.Sequential(*sq2)
        self.blockout = FuisonBlock(d_model, d_ff, n_head)
        

        
    def forward(self, x):# data: batch, ..., d_model
        data_1, data_2 = x
        data_1, data_2 = self.faltten(data_1), self.faltten(data_2) #展平中间维度
        data_1, data_2 = data_1.permute(0, 2, 1), data_1.permute(0, 2, 1)
        data_1, data_2 = self.pe(data_1) + data_1, self.pe(data_2) + data_2 #位置编码
        for fusionblock1, fusionblock2 in zip(self.fusionseq1, self.fusionseq2):
            data_1 = fusionblock1(data_1, data_2)
            data_2 = fusionblock2(data_2, data_1)
        self.blockout(data_1, data_2)
        
        data_1, data_2 = data_1.permute(0, 2, 1), data_1.permute(0, 2, 1)
        return data_1

class FuisonBlock(nn.Module):
    def __init__(self, d_model, d_ff=None, n_head=1):
        super().__init__()

        d_ff = d_model if d_ff is None else d_ff

        self.quary_sa = nn.Linear(d_model, d_model)
        self.key_sa = nn.Linear(d_model, d_model)
        self.value_sa = nn.Linear(d_model, d_model)

        self.ma_sa = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_sa = nn.LayerNorm(d_model)

        self.quary_fa = nn.Linear(d_model, d_model)
        self.key_fa = nn.Linear(d_model, d_model)
        self.value_fa = nn.Linear(d_model, d_model)

        self.ma_fa = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_fa = nn.LayerNorm(d_model)

        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ln_ff = nn.LayerNorm(d_model)

        for m in (
            self.quary_sa, 
            self.key_sa, 
            self.value_sa, 
            self.quary_fa, 
            self.key_fa, 
            self.value_fa,
            self.ff1,
            self.ff2,
            ):
            xavier_uniform_(m.weight.data)
            constant_(m.bias.data, 0.0)


    def forward(self, data_q, data_kv):
        # self attention
        self_attn = self.ma_sa(self.quary_sa(data_q), self.key_sa(data_q), self.value_sa(data_q))[0] + data_q
        out_sa = self.ln_sa(self_attn)

        # fusion attention
        fusion_attn = self.ma_fa(self.quary_fa(out_sa), self.key_fa(data_kv), self.value_fa(data_kv))[0] + out_sa
        out_fa = self.ln_fa(fusion_attn)

        #feed forward
        ff = self.ff2(self.ff1(out_fa)) + out_fa
        return self.ln_ff(ff)
    
class FusionConcatInput(nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()
    def forward(self, x):
        x = [y for y in x]
        return torch.cat(x, dim=self.dim)
 
class FusionSplitResult(nn.Module):
    def __init__(self, n_start, n_end):
        super().__init__()

        self.n_start = n_start
        self.n_end = n_end
        n = int(math.sqrt(n_end - n_start))
        self.uf = nn.Unflatten(-1, torch.Size([n, n]))
        
    
    def forward(self, x):
        r = self.uf(x[:,:,self.n_start:self.n_end])
        return r

class FusionLinear(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.l = nn.Linear(c1, c2)
        xavier_uniform_(self.l.weight.data)
        constant_(self.l.bias.data, 0.0)

    def forward(self, x:torch.Tensor):
        x = x.transpose(1, -1)
        x = self.l(x)
        return x.transpose(1, -1)
