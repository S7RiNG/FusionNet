import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D


class Lidar_group(nn.Module):
    def __init__(self, shape, depth=9):
        super().__init__()
        self.shape = shape
        self.depth = depth

    def forward(self, data:torch.Tensor):  # data *, (x, y, *）
        # batchsize, group x, y, stack_depth, point
        groupsize = (data.shape[0], self.shape, self.shape, self.depth, data.shape[-1])
        group = torch.zeros(groupsize)
        scale = torch.ones(data.shape[-1])
        scale[0] = self.shape
        scale[1] = self.shape
        torch.asarray(scale, dtype=torch.float)
        data = data * scale
        zero_data = torch.zeros(data.shape[-1])

        for ins_n in range(data.shape[0]):
            for pt in data[ins_n]:
                if torch.equal(pt, zero_data):
                    continue
                x, y = int(pt[0]), int(pt[1])
                for d in range(self.depth):
                    if torch.equal(group[ins_n][x][y][d], zero_data):
                        pt_group = pt
                        pt_group[0] -= x
                        pt_group[1] -= y
                        group[ins_n][x][y][d] = pt_group
                        break
                    else:
                        if d == self.depth - 1:
                            print("warning: point out of stack.")
                        continue
        group_stack = group.reshape(data.shape[0], self.shape, self.shape, -1)
        return group_stack

class Lidar_microattn(nn.Module):
    def __init__(self, d_model):
        self.attn = nn.MultiheadAttention(d_model, 1, 0.1)
        self.pos = PositionalEncoding2D(d_model)
        super().__init__()

    def forward(self, rgb:torch.Tensor, pt_group:torch.Tensor):
        assert(rgb.shape[:2] != pt_group.shape[:2])
        o = torch.zeros_like(rgb)
        w, h = rgb.shape[:2]
        for x in range(w - 3):
            for y in range(h - 3):
                d_rgb = rgb[x:x + 3, y:y + 3]
                d_pt = pt_group[x:x + 3, y:y + 3]
                d_rgb = d_rgb.reshape((3 * 3, -1))
                d_pt = d_pt.reshape((3 * 3, -1))
                o = self.attn(d_rgb, d_pt, d_pt)

        pass


data = [
    [0.1, 0.2, 1],
    [0.11, 0.1, 2],
    [0.12, 0.11, 5],
    [0.22, 0.4, 3],
    [0.21, 0.5, 4]
]

test_pt = torch.rand((2, 100000, 4))
test_rgb = torch.rand((2, 32, 64, 64))

group = Lidar_group(64)
grp = group.forward(test_pt)
print(grp.shape)

