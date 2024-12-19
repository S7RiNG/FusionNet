import torch.nn as nn
import torch
import math
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, PositionalEncoding2D, PositionalEncodingPermute2D
from torch.nn.init import constant_, xavier_uniform_
from .block import C2f 

__all__ = ('FuisonBlock', "FusionConcatInput", 'FusionSequence', 'FusionSplitResult', 'FusionLinear')


class FusionSequence(nn.Module):
    def __init__(self, d_model, repeat=1, n_head=1, d_ff=None):
        super().__init__()
        self.d_model = d_model
        d_ff = d_model if d_ff is None else d_ff
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
        # data_1, data_2 = self.faltten(data_1), self.faltten(data_2) #展平中间维度
        data_1, data_2 = data_1.permute(0, 2, 1), data_1.permute(0, 2, 1)
        data_1, data_2 = self.pe(data_1) + data_1, self.pe(data_2) + data_2 #位置编码
        for fusionblock1, fusionblock2 in zip(self.fusionseq1, self.fusionseq2):
            data_1 = fusionblock1(data_1, data_2)
            data_2 = fusionblock2(data_2, data_1)
        data = self.blockout(data_1, data_2)
        
        data = data.permute(0, 2, 1)
        return data

class FuisonBlock(nn.Module):
    def __init__(self, d_model, d_ff=None, n_head=1):
        super().__init__()

        d_ff = d_model if d_ff is None else d_ff

        self.quary_fa = nn.Linear(d_model, d_model)
        self.key_fa = nn.Linear(d_model, d_model)
        self.value_fa = nn.Linear(d_model, d_model)

        self.ma_fa = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_fa = nn.LayerNorm(d_model)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ln_ff = nn.LayerNorm(d_model)


        for m in (
            self.quary_fa, 
            self.key_fa, 
            self.value_fa,
            self.ff1,
            self.ff2,
            ):
            xavier_uniform_(m.weight.data)
            constant_(m.bias.data, 0.0)


    def forward(self, data_q, data_kv):
        # fusion attention
        fusion_attn = self.ma_fa(self.quary_fa(data_q), self.key_fa(data_kv), self.value_fa(data_kv))[0] + data_q
        out_fa = self.ln_fa(fusion_attn)

        # feed forward
        ff = self.ff2(self.dropout(self.relu(self.ff1(out_fa))))
        return self.ln_ff(ff + out_fa)
    
class FusionSequence_SA(nn.Module):
    def __init__(self, d_model, repeat=1, n_head=1, d_ff=None):
        super().__init__()
        self.d_model = d_model
        d_ff = d_model if d_ff is None else d_ff
        sq1, sq2 = [], []

        for _ in range(repeat):
            sq1.append(FuisonBlock_SA(d_model, d_ff, n_head))
            sq2.append(FuisonBlock_SA(d_model, d_ff, n_head))
        self.fusionseq1 = nn.Sequential(*sq1)
        self.fusionseq2 = nn.Sequential(*sq2)
        self.blockout = FuisonBlock(d_model, d_ff, n_head)
        
    def forward(self, x):# data: batch, ..., d_model
        data_1, data_2 = x
        data_1, data_2 = data_1.permute(0, 2, 1), data_2.permute(0, 2, 1)
        for fusionblock1, fusionblock2 in zip(self.fusionseq1, self.fusionseq2):
            data_1 = fusionblock1(data_1, data_2)
            data_2 = fusionblock2(data_2, data_1)
        data = self.blockout(data_1, data_2)
        
        data = data.permute(0, 2, 1)
        return data

class FuisonBlock_SA(nn.Module):
    def __init__(self, d_model, d_ff=None, n_head=1):
        super().__init__()

        d_ff = d_model if d_ff is None else d_ff

        self.ma_fa = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_fa = nn.LayerNorm(d_model)

        self.ma_sa = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_sa = nn.LayerNorm(d_model)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ln_ff = nn.LayerNorm(d_model)


        for m in (
            self.ff1,
            self.ff2,
            ):
            xavier_uniform_(m.weight.data)
            constant_(m.bias.data, 0.0)


    def forward(self, data_q, data_kv):
        # fusion attention
        fusion_attn = self.ma_fa(data_q, data_kv,data_kv)[0] + data_q
        out_fa = self.ln_fa(fusion_attn)

        self_attn = self.ma_sa(out_fa, out_fa, out_fa)[0] + out_fa
        out_sa = self.ln_fa(self_attn)

        # feed forward
        ff = self.ff2(self.dropout(self.relu(self.ff1(out_sa))))
        return self.ln_ff(ff + out_sa)
    
class FuisonBlock_FC(nn.Module):
    def __init__(self, d_model, d_ff=None, n_head=1):
        super().__init__()

        d_ff = d_model if d_ff is None else d_ff

        self.ma_fa = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=0.1)
        self.ln_fa = nn.LayerNorm(d_model)

        self.ma_sa = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=0.1)
        self.ln_sa = nn.LayerNorm(d_model)

        self.relu = nn.ReLU()
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ln_ff = nn.LayerNorm(d_model)


        for m in (
            self.ff1,
            self.ff2,
            ):
            xavier_uniform_(m.weight.data)
            constant_(m.bias.data, 0.0)


    def forward(self, data_q, data_kv):
        # fusion attention
        fusion_attn = self.ma_fa(data_q, data_kv,data_kv)[0] + data_q
        out_fa = self.ln_fa(fusion_attn)

        self_attn = self.ma_sa(out_fa, out_fa, out_fa)[0] + out_fa
        out_sa = self.ln_fa(self_attn)

        # feed forward
        ff = self.relu(self.ff2(self.relu(self.ff1(out_sa))))
        return self.ln_ff(ff + out_sa)


class FuisonBlock_CSP(nn.Module):
    def __init__(self, d_model, d_kv=None, n_cat=3, n_head=1, dropout=0):
        super().__init__()

        d_kv = d_model if d_kv is None else d_kv

        self.ma_fa = nn.MultiheadAttention(d_model, n_head, batch_first=True, kdim=d_kv, vdim=d_kv, dropout=dropout)
        self.ln_fa = nn.LayerNorm(d_model)

        self.ma_sa = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.ln_sa = nn.LayerNorm(d_model)

        self.csp = nn.Sequential(*[C2f(d_model, d_model) for _ in range(n_cat)])
        self.cat = FusionConcatInput(-1)

    def forward(self, data_q, data_kv):
        # fusion attention
        fusion_attn = self.ma_fa(data_q, data_kv,data_kv)[0] + data_q
        out_fa = self.ln_fa(fusion_attn)

        self_attn = self.ma_sa(out_fa, out_fa, out_fa)[0] + out_fa
        out_sa = self.ln_fa(self_attn)
        
        out_sa = out_sa.permute(0, 2, 1)
        out_split = self._split(out_sa)
        out_csp = [csp(x) for x, csp in zip(out_split, self.csp)]
        out = self.cat(out_csp).permute(0, 2, 1)
        return out
    
    def _split(self, x:torch.Tensor) -> list[torch.Tensor]:
        n_cat = len(self.csp)
        nx = x.shape[-1]
        n_list = [(4 ** level) for level in range(n_cat)]
        n_list = [sum(n_list[0:(i + 1)]) for i in range(n_cat)]
        base = nx // n_list[-1]
        edge = [n * base for n in n_list]
        edge.insert(0, 0)
        split = [x[:, :, edge[i]:edge[i+1]] for i in range(n_cat)]
        return [s.unflatten(-1, [int(math.sqrt(s.shape[-1]))] * 2) for s in split]

class FusionConcatInput(nn.Module):
    def __init__(self, dim, ):
        self.dim = dim
        super().__init__()
        self.faltten = nn.Flatten(2, -1)

    def forward(self, x):
        x = [self.faltten(y) for y in x]
        return torch.cat(x, dim=self.dim)
    
class FusionConcatInput_PE(nn.Module):
    def __init__(self, dim, ch):
        self.dim = dim
        super().__init__()
        # self.faltten = nn.Flatten(2, -1)
        self.pe2d = PositionalEncodingPermute2D(ch)
        self.pe1d = PositionalEncodingPermute1D(ch)

    def forward(self, x:list[torch.Tensor]):
        x = [y + self.pe2d(y) for y in x]
        x = [y.flatten(2, -1) for y in x]
        x = torch.cat(x, dim=self.dim)
        return x + self.pe1d(x)
 
class FusionSplitResult(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.level = level
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        length = x.shape[-1]
        base = int(length / 21)
        edge = [0, base, base * 5, base * 21]
        n_end = edge[self.level]
        n_start = edge[self.level - 1]
        side = int(math.sqrt(n_end - n_start))
        
        r = x[:,:,n_start:n_end]
        r = r.unflatten(-1, (side, side))
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
    
class FusionConv1d(FusionLinear):
    pass

class FusionExtend1d(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        scale = c2 // c1
        c_in1 = int(math.sqrt(scale // 2)) * c1
        c_in2 = c2 // 2
        c_in3 = c2 - c_in2

        self.cv1 = nn.Conv1d(c1, c_in1, 1, 1)
        self.cv2 = nn.Conv1d(c_in1, c_in2, 1, 1)
        self.cv3 = nn.Conv1d(c1, c_in3, 1, 1)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(c2)

    def forward(self, x:torch.Tensor):
        y1 = self.act(self.cv2(self.act(self.cv1(x))))
        y2 = self.act(self.cv3(x))
        return self.bn1(torch.cat([y1, y2], 1))

class FusionPointAttenetion(nn.Module):
    def __init__(self, d_model, size, n_head=1, dropout=0):
        super().__init__()
        self.pe2d = PositionalEncoding2D(d_model)
        self.size = [*size, d_model]
        self.fb = FuisonBlock_CSP(d_model, n_cat=1, n_head=n_head, dropout=dropout)
        # self.ma = nn.MultiheadAttention(d_model, num_heads=n_head, batch_first=True)
        # self.ln = nn.LayerNorm(d_model)
        
        
    def forward(self, x:torch.Tensor):
        b = x.shape[0]
        shape = (b, *self.size)
        quray = torch.Tensor(self.pe2d(torch.zeros(shape, device=x.device, dtype=x.dtype))).flatten(1,2)
        return self.fb(quray, x)
        # attn = self.ma(quray, x, x)[0] + quray
        # return self.ln(attn)

    
class FusionImageLidar(nn.Module):
    def __init__(self, d_img, d_ldr, repeat=1, n_head=1, dropout=0):
        super().__init__()

        self.pa = FusionPointAttenetion(d_ldr, [20, 20], n_head=n_head, dropout=dropout)

        self.seq_img = nn.Sequential()
        self.seq_ldr = nn.Sequential()

        for _ in range(repeat):
            self.seq_img.append(FuisonBlock_CSP(d_img, d_ldr, n_head=n_head, n_cat=3, dropout=dropout))
            self.seq_ldr.append(FuisonBlock_CSP(d_ldr, d_img, n_head=n_head, n_cat=1, dropout=dropout))

        self.blockout = FuisonBlock_CSP(d_img, d_ldr, n_head=n_head, dropout=dropout)
        
    def forward(self, x):# data: batch, d_model, n
        data_1, data_2 = x
        data_1, data_2 = data_1.permute(0, 2, 1), data_2.permute(0, 2, 1)

        data_2 = self.pa(data_2)

        for fusionblock1, fusionblock2 in zip(self.seq_img, self.seq_ldr):
            data_1 = fusionblock1(data_1, data_2)
            data_2 = fusionblock2(data_2, data_1)

        data = self.blockout(data_1, data_2)

        data = data.permute(0, 2, 1)
        return data

