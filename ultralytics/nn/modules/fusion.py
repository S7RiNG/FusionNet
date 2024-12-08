import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D

__all__ = ('FuisonBlock', 'FusionNet')


class FuisonBlock(nn.Module):
    def __init__(self, d_model, d_ff=2048, n_head=1):
        super().__init__()

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

class FusionNet(nn.Module):
    def __init__(self, d_model, repeat=1, d_ff=2048, n_head=1):
        super().__init__()

        self.faltten = nn.Flatten(1, -2)        
        self.pe = PositionalEncoding1D(d_model)
        self.fbl1, self.fbl2 = [], []

        for _ in range(repeat):
            self.fbl1.append(FuisonBlock(d_model, d_ff, n_head))
            self.fbl2.append(FuisonBlock(d_model, d_ff, n_head))
        
    def forward(self, data_1, data_2):# data: batch, ..., d_model
        _data_1, _data_2 = self.faltten(data_1), self.faltten(data_2) #展平中间维度
        _data_1, _data_2 = self.pe(_data_1) + _data_1, self.pe(_data_2) + _data_2 #位置编码
        for fb1, fb2 in zip(self.fbl1, self.fbl2):
            _data_1 = fb1(_data_1, _data_2)
            _data_2 = fb2(_data_2, _data_1)
        return _data_1, _data_2
    

    