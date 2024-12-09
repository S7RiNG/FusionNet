from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import functional as f
import cv2

def read_lidarmap(path) -> Tensor:
    return f.to_tensor(cv2.imread(path))

def read_lidarpoint(path) -> Tensor:
    return torch.from_numpy(np.load(path))

def read_combo(img_path) -> Tensor:
    path = Path(img_path)
    path_map = path.parent/'..'/'maps'/path.name
    path_point = Path.with_suffix((path.parent/'..'/'points'/path.name), '.npy') 

    im = f.to_tensor(cv2.imread(path))
    map = read_lidarmap(path_map)
    point = read_lidarpoint(path_point)
    return torch.cat([im, map]).permute(1,2,0).numpy(), point

#for test
if __name__ == '__main__':
    path = r'E:\Dataset\kitti\yolo_fusion\train\images\000000.png'
    r = read_combo(path)
    print([y.shape for y in r]) #(C, H, W), (N, Pt)