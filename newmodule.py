# import torch
# import torch.nn as nn
# from positional_encodings.torch_encodings import PositionalEncoding2D


# class Lidar_group(nn.Module):
#     def __init__(self, shape, depth=8):
#         super().__init__()
#         self.shape = shape
#         self.depth = depth

#     def forward(self, data:torch.Tensor):  # data *, (x, y, *）
#         # batchsize, group x, y, stack_depth, point
#         groupsize = (data.shape[0], self.shape, self.shape, self.depth, data.shape[-1])
#         group = torch.zeros(groupsize)
#         scale = torch.ones(data.shape[-1])
#         scale[0] = self.shape
#         scale[1] = self.shape
#         torch.asarray(scale, dtype=torch.float)
#         data = data * scale
#         zero_data = torch.zeros(data.shape[-1])

#         for ins_n in range(data.shape[0]):
#             for pt in data[ins_n]:
#                 if torch.equal(pt, zero_data):
#                     continue
#                 x, y = int(pt[0]), int(pt[1])
#                 for d in range(self.depth):
#                     if torch.equal(group[ins_n][x][y][d], zero_data):
#                         pt_group = pt
#                         pt_group[0] -= x
#                         pt_group[1] -= y
#                         group[ins_n][x][y][d] = pt_group
#                         break
#                     else:
#                         if d == self.depth - 1:
#                             print("warning: point out of stack at", x, y)
#                         continue
#         group_stack = group.reshape(data.shape[0], self.shape, self.shape, -1).permute(0,3,1,2) #n, c, w, h
#         return group_stack

# class Lidar_microattn(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(d_model, 1, 0.1, batch_first=True)
#         self.pos = PositionalEncoding2D(d_model)
#         self.cv = nn.Conv2d(d_model * 9, d_model, 3, 1, 1)
        

#     def forward(self, rgb:torch.Tensor, pt_group:torch.Tensor):
#         shape = torch.asarray(rgb.shape, dtype=torch.int)
#         shape[1] *= 9
#         o = torch.zeros(list(shape)).permute(2,3,0,1)
#         rgb = nn.functional.pad(rgb, [1,1,1,1], 'replicate')
#         pt_group = nn.functional.pad(pt_group, [1,1,1,1], 'constant')
#         w, h = rgb.shape[2:4]

#         for x in range(w - 2):
#             for y in range(h - 2):
#                 d_rgb = rgb[:,:,x:x + 3, y:y + 3]
#                 d_rgb += self.pos(d_rgb)
#                 d_rgb = d_rgb.reshape(d_rgb.shape[0], d_rgb.shape[1], -1).transpose(-2, -1)

#                 d_pt = pt_group[:,:,x:x + 3, y:y + 3]
#                 d_pt += self.pos(d_pt)
#                 d_pt = d_pt.reshape(d_pt.shape[0], d_pt.shape[1], -1).transpose(-2, -1)

#                 o_xy = self.attn(d_rgb, d_pt, d_pt)[0]
#                 o_xy = o_xy.reshape(o_xy.shape[0], -1)
#                 o[x][y] = o_xy
        
#         o = self.cv(o.permute(2,3,0,1))
#         return o
    
# class Lidar_microattn2(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.k = k = 3
#         self.attn = nn.MultiheadAttention(d_model, 1, 0.1, batch_first=True)
#         self.pos = PositionalEncoding2D(d_model)
#         self.cv_ff = nn.Conv2d(d_model * k * k, d_model * k * k * 8, 1, 1, 1)
#         self.cv_fit = nn.Conv2d(d_model * k * k * 8, d_model, 3, 1, 1)
        

#     def forward(self, rgb:torch.Tensor, pt_group:torch.Tensor):
#         k = self.k
#         shape = torch.asarray(rgb.shape, dtype=torch.int)

#         shape_buf = shape * torch.asarray([1, 2, k, k])
#         buf = torch.zeros(list(shape_buf))

#         pad = k//2
#         rgb = nn.functional.pad(rgb, [pad,pad,pad,pad], 'replicate')
#         pt_group = nn.functional.pad(pt_group, [pad,pad,pad,pad], 'replicate')
#         n, c, w, h = shape

#         buf[:, :c, :w + 2, :h + 2] = rgb
#         buf[:, c:, :w + 2, :h + 2] = pt_group
#         for x in range(w, 0, -1):
#             x -= 1
#             buf[:, :, k*x:k*(x+1), :] = buf[:, :, x:x+k, :] * (x+1)
#         for y in range(h, 0, -1):
#             y -= 1
#             buf[:, :, :, k*y:k*(y+1)] = buf[:, :, :, y:y+k] * (y+1)
#         buf = torch.reshape(buf, (n, 2, c, w, k, h, k)).permute(1, 0, 3, 5, 4, 6, 2).reshape(2, n*w*h, k, k, c)
        

#         for i in range(2):
#             pos = self.pos(buf[i])
#             buf[i] += pos
#         buf = buf.reshape(2, n*w*h, k*k, c)
#         buf_rgb, buf_pt = buf[0], buf[1]
        
#         assert c == self.attn.embed_dim
#         o = self.attn(buf_rgb, buf_pt, buf_pt)[0]
#         o = torch.reshape(o, (n, w, h, k * k * c)).permute(0, 3, 1, 2)
#         o = self.cv_ff(self.cv_fit(o))
#         return o


# data = [
#     [0.1, 0.2, 1],
#     [0.11, 0.1, 2],
#     [0.12, 0.11, 5],
#     [0.22, 0.4, 3],
#     [0.21, 0.5, 4]
# ]

# test_pt = torch.rand((2, 1000, 4))
# test_rgb = torch.ones((1, 32, 4, 4))
# test_rgb[:,1] *= 5
# test_rgb[:,2] *= 3
# test_pt = torch.ones((1, 32, 4, 4))
# # group = Lidar_group(8)
# # grp = group.forward(test_pt)
# # print(grp.shape)

# ma = Lidar_microattn2(32)
# o = ma(test_rgb, test_pt)
# print(o.shape)

from ultralytics.nn.modules.fusion import Lidar_HelfMaxpool
import torch

pool = Lidar_HelfMaxpool()

d1 = torch.rand(1, 3, 7, 7)
d2 = torch.rand(1, 3, 7, 7)

d = pool(d1, d2)
print(d1)
print(d2)
print(d)