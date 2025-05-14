import numpy as np
import torch.nn as nn
import torch
import math
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, PositionalEncoding2D, PositionalEncodingPermute2D
from torch.nn.init import constant_, xavier_uniform_
from .block import C2f, SPPF
from .Fusion_SwinTransformer import BasicLayer_Cross as Swin_BasicLayer_Cross, BasicLayer as Swin_BasicLayer, PatchMerging as Swin_PatchMerging, PatchEmbed as Swin_PatchEmbed

import time

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

class FusionLidarAttenetionInStage(nn.Module):
    def __init__(self, d_model, n_head=1, dropout=0):
        super().__init__()
        self.pe2d = PositionalEncoding2D(d_model)
        self.d_model = d_model
        self.fb = FusionLidarAttenetion(d_model, n_head=n_head, dropout=dropout)
        # self.ma = nn.MultiheadAttention(d_model, num_heads=n_head, batch_first=True)
        # self.ln = nn.LayerNorm(d_model)
        
        
    def forward(self, x:torch.Tensor, size):
        b = x.shape[0]
        shape = (b, *size, self.d_model)
        quray = torch.Tensor(self.pe2d(torch.zeros(shape, device=x.device, dtype=x.dtype))).flatten(1,2)
        return self.fb(quray, x, size)
        # attn = self.ma(quray, x, x)[0] + quray
        # return self.ln(attn)

class FusionLidarAttenetion(nn.Module):
    def __init__(self, d_model, d_kv=None, n_head=1, dropout=0):
        super().__init__()

        d_kv = d_model if d_kv is None else d_kv

        self.ma_fa = nn.MultiheadAttention(d_model, n_head, batch_first=True, kdim=d_kv, vdim=d_kv, dropout=dropout)
        self.ln_fa = nn.LayerNorm(d_model)

        self.ma_sa = nn.MultiheadAttention(d_model, n_head, batch_first=True, dropout=dropout)
        self.ln_sa = nn.LayerNorm(d_model)

        self.csp = C2f(d_model, d_model)

    def forward(self, data_q, data_kv, size):
        # fusion attention
        fusion_attn = self.ma_fa(data_q, data_kv,data_kv)[0] + data_q
        out_fa = self.ln_fa(fusion_attn)

        self_attn = self.ma_sa(out_fa, out_fa, out_fa)[0] + out_fa
        out_sa = self.ln_fa(self_attn)
        
        out_sa = out_sa.permute(0, 2, 1)
        out_uf = out_sa.unflatten(-1, size)
        out_csp = self.csp(out_uf)
        out = out_csp.flatten(2, -1)
        return out.permute(0, 2, 1)
    
class FusionImageLidar(nn.Module):
    def __init__(self, d_img, d_ldr, repeat=1, n_head=1, dropout=0):
        super().__init__()

        self.pa = FusionLidarAttenetionInStage(d_ldr, [20, 20], n_head=n_head, dropout=dropout)

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

class FusionLidar(nn.Module):
    def __init__(self, c1, c2, repeat=1, n_head=1, dropout=0, stride=1):
        super().__init__()
        self.stride = stride
        self.pa = FusionLidarAttenetionInStage(c1, n_head=n_head, dropout=dropout)

        self.seq = nn.Sequential()

        for _ in range(repeat):
            self.seq.append(FusionLidarAttenetion(c1, c1, n_head=n_head, dropout=dropout))

        self.sppf = SPPF(c1, c2)
        
    def forward(self, x, rgbsize):# data: batch, d_model, n
        
        size = [side // self.stride for side in rgbsize]
        data = x.permute(0, 2, 1)
        out = self.pa(data, size)
        
        for layer in self.seq:
            out = layer(out, data, size)
        
        out = out.permute(0, 2, 1)
        out = out.unflatten(-1, size)
        out = self.sppf(out)
        return out
    
class FusionLidar_Pyrmid(FusionLidar):
    def __init__(self, c1, c2, repeat=1, n_head=1, dropout=0, stride=1):
        super(FusionLidar, self).__init__()
        self.stride = stride
        self.repeat = repeat
        self.pa = FusionLidarAttenetionInStage(c1, n_head=n_head, dropout=dropout)

        self.seq_fa = nn.Sequential()
        self.seq_cv = nn.Sequential()

        for _ in range(repeat):
            self.seq_cv.append(nn.Conv2d(c1, c1, 3, 2, 1))
            self.seq_fa.append(FusionLidarAttenetion(c1, c1, n_head=n_head, dropout=dropout))

        self.sppf = SPPF(c1, c2)
    
    def forward(self, x, rgbsize):# data: batch, d_model, n
        
        size = [side // self.stride for side in rgbsize]
        size_fa = [side * 2 ** self.repeat for side in size]
        data = x.permute(0, 2, 1)
        out = self.pa(data, size_fa)
        
        for cv, fa in zip(self.seq_cv, self.seq_fa):
            out = out.permute(0, 2, 1)
            out = out.unflatten(-1, size_fa)
            out = cv(out)
            out = out.flatten(2, -1)
            out = out.permute(0, 2, 1)
            size_fa = [side // 2 for side in size_fa]
            out = fa(out, data, size_fa)
            

        
        out = out.permute(0, 2, 1)
        out = out.unflatten(-1, size)
        out = self.sppf(out)
        return out

class Lidar_group(nn.Module):
    def __init__(self, shape, depth=4):
        super().__init__()
        self.shape = shape
        self.depth = depth

    def forward(self, data:torch.Tensor, rgbsize):  # data b , (x, y, *), n
        # batchsize, group x, y, stack_depth, point
        w, h = rgbsize
        dev = data.device
        data = data.to(torch.device('cpu'))

        data = data.transpose(1, 2)
        groupsize = (data.shape[0], w, h, self.depth, data.shape[-1])
        group = torch.zeros(groupsize)
        scale = torch.ones(data.shape[-1])
        scale[0] = w
        scale[1] = h
        torch.asarray(scale, dtype=torch.float)
        data = data * scale
        zero_data = torch.zeros(data.shape[-1])

        for ins_n in range(data.shape[0]):
            for pt in data[ins_n]:
                if torch.equal(pt, zero_data):
                    continue
                x, y = int(pt[0]), int(pt[1])

                if x < 0 or x >= w or y < 0 or y >= h:
                    continue

                for d in range(self.depth):
                    if torch.equal(group[ins_n][x][y][d], zero_data):
                        pt_group = pt
                        pt_group[0] -= (x + 0.5)
                        pt_group[1] -= (y + 0.5)
                        group[ins_n][x][y][d] = pt_group
                        break
                    else:
                        if d == self.depth - 1:
                            print("warning: point out of stack at", x, y)
                        continue
        group = group.to(dev)
        group_stack = group.reshape(data.shape[0], w, h, -1).permute(0,3,1,2) #n, c, w, h
        return group_stack

class Lidar_PositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pos = PositionalEncoding2D(d_model)
    
    def forward(self, x):
        return x + self.pos(x)
    
# class Lidar_PositionalEncoding2D(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.pos = PositionalEncoding2D(d_model)
#         self.add = torch.zeros((1,1,1,1))
    
#     def forward(self, x:torch.Tensor):
#         n, c, w, h = x.shape
#         na, ca, wa, wh = self.add.shape
#         if n != na or c != ca or w!=wa or h != wh or x.device != self.add.device or x.dtype != self.add.dtype:
#             self.add = self.pos(x)
#         return x + self.add

class Lidar_microattn(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.k = k = 3
        self.attn = nn.MultiheadAttention(d_model, 1, 0.1, batch_first=True)
        self.pos = PositionalEncoding2D(d_model)
        self.bn_attn = nn.BatchNorm2d(d_model*k*k)
        self.cv_ff = nn.Conv2d(in_channels=d_model*k*k, out_channels=d_model*k*k, kernel_size=1, stride=1, padding=0)
        self.cv_fit = nn.Conv2d(in_channels=d_model*k*k, out_channels=d_model, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.bn_ff = nn.BatchNorm2d(d_model)

    def forward(self, x:list[torch.Tensor], rgbsize):
        data_q, data_kv = x
        k = self.k
        shape = torch.asarray(data_q.shape, dtype=torch.int)

        shape_buf = shape * torch.asarray([1, 2, k, k])
        buf = torch.zeros(list(shape_buf), device=data_q.device, dtype=data_q.dtype)

        pad = k//2
        data_q = nn.functional.pad(data_q, [pad,pad,pad,pad], 'replicate')
        data_kv = nn.functional.pad(data_kv, [pad,pad,pad,pad], 'constant')
        n, c, w, h = shape

        data=torch.cat((data_q, data_kv), 1)
        for dx in range(k):
            for dy in range(k):
                buf[:,:, dx : k*w+dx : k, dy : k*h + dy : k] = data[:,:, dx:w+dx,dy:h+dy]

        # buf[:, :c, :w + 2, :h + 2] = rgb
        # buf[:, c:, :w + 2, :h + 2] = pt_group
        # for x in range(w, 0, -1):
        #     x -= 1
        #     buf[:, :, k*x:k*(x+1), :] = buf[:, :, x:x+k, :]
        # for y in range(h, 0, -1):
        #     y -= 1
        #     buf[:, :, :, k*y:k*(y+1)] = buf[:, :, :, y:y+k]

        # if (torch.equal(buf, buf2)) == False: raise

        buf = torch.reshape(buf, (n, 2, c, w, k, h, k)).permute(1, 0, 3, 5, 4, 6, 2).reshape(2, n*w*h, k, k, c)
        
        # for i in range(2):
        #     pos = self.pos(buf[i])
        #     buf[i] += pos
        buf = buf.reshape(2, n*w*h, k*k, c)
        buf_q, buf_kv = buf[0], buf[1]
        
        assert c == self.attn.embed_dim
        o = self.attn(buf_q, buf_kv, buf_kv)[0] + buf_kv
        o = torch.reshape(o, (n, w, h, k * k * c)).permute(0, 3, 1, 2)
        o = self.bn_attn(o)
        o = self.cv_fit(self.relu(self.cv_ff(o)) + o)
        o = self.bn_ff(o)
        return o
    
class Lidar_microattn2(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.k = k = 3
        self.attn = nn.MultiheadAttention(d_model, 1, 0.1, batch_first=True)
        self.pos = PositionalEncoding2D(d_model)
        self.cv_ff = nn.Conv2d(d_model * k * k, d_model * k * k * 8, 1, 1, 1)
        self.cv_fit = nn.Conv2d(d_model * k * k * 8, d_model, 3, 1, 1)
        

    def forward(self, rgb:torch.Tensor, pt_group:torch.Tensor):
        k = self.k
        shape = torch.asarray(rgb.shape, dtype=torch.int)

        shape_buf = shape * torch.asarray([1, 2, k, k])
        buf = torch.zeros(list(shape_buf))

        pad = k//2
        rgb = nn.functional.pad(rgb, [pad,pad,pad,pad], 'replicate')
        pt_group = nn.functional.pad(pt_group, [pad,pad,pad,pad], 'replicate')
        n, c, w, h = shape

        buf[:, :c, :w + 2, :h + 2] = rgb
        buf[:, c:, :w + 2, :h + 2] = pt_group
        for x in range(w, 0, -1):
            x -= 1
            buf[:, :, k*x:k*(x+1), :] = buf[:, :, x:x+k, :] * (x+1)
        for y in range(h, 0, -1):
            y -= 1
            buf[:, :, :, k*y:k*(y+1)] = buf[:, :, :, y:y+k] * (y+1)
        buf = torch.reshape(buf, (n, 2, c, w, k, h, k)).permute(1, 0, 3, 5, 4, 6, 2).reshape(2, n*w*h, k, k, c)
        

        for i in range(2):
            pos = self.pos(buf[i])
            buf[i] += pos
        buf = buf.reshape(2, n*w*h, k*k, c)
        buf_rgb, buf_pt = buf[0], buf[1]
        
        assert c == self.attn.embed_dim
        o = self.attn(buf_rgb, buf_pt, buf_pt)[0]
        o = torch.reshape(o, (n, w, h, k * k * c)).permute(0, 3, 1, 2)
        o = self.cv_ff(self.cv_fit(o))
        return o
    
class Lidar_HelfMaxpool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d((2, 1), (2, 1))

    def forward(self, x:list[torch.Tensor]):
        data1, data2 = x
        n, c, w, h = data1.shape
        data = torch.stack([data1, data2], 3).reshape(n, c, w*2, h)
        data = self.pool(data)
        return data
    
class Lidar_Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:list[torch.Tensor]):
        data1, data2 = x
        return data1 + data2
    


class SwinEmbed(nn.Module):
    def __init__(self, d_in, d_out, patch_size):
        super().__init__()
        self.emb = Swin_PatchEmbed(in_c=d_in, embed_dim=d_out, patch_size=patch_size, norm_layer=nn.LayerNorm)
    
    def forward(self, x):
        x, W, H = self.emb(x)
        return torch.unflatten(x.permute(0,2,1), -1, (W, H))
    
class SwinBlock(nn.Module):
    def __init__(self, d_model, is_downsample):
        super().__init__()
        self.swin_layear = Swin_BasicLayer(d_model, 2, 8, 7, downsample=Swin_PatchMerging if is_downsample else None)
    
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        B, C, W, H = X.shape
        X = X.flatten(2).permute(0,2,1) #B;L;C
        res, W, H = self.swin_layear(X, W, H)
        return torch.unflatten(res.permute(0,2,1), -1, (W, H))

class SwinBlock_Cross(nn.Module):
    def __init__(self, d_model, is_downsample):
        super().__init__()
        self.swin_layear = Swin_BasicLayer_Cross(d_model, 2, 8, 7, downsample=Swin_PatchMerging if is_downsample else None)
    
    def forward(self, X:list[torch.Tensor]) -> torch.Tensor:
        data_q, data_kv = X
        B, C, W, H = data_q.shape
        data_q = data_q.flatten(2).permute(0,2,1) #B;L;C
        data_kv = data_kv.flatten(2).permute(0,2,1) #B;L;C
        res, W, H = self.swin_layear(data_q, data_kv, W, H)

        return torch.unflatten(res.permute(0,2,1), -1, (W, H))
