import sys
from ultralytics.models.fusion import FusionNet
from ultralytics.models.yolo import YOLO
from ultralytics.nn.modules.fusion import FusionSequence, FusionLinear
import torchinfo 
import torch
from torch import nn
from ultralytics.data.lidar import LetterBox_LiDAR, LiDAR_norm
import numpy as np

if __name__ == "__main__":

    if sys.platform == 'win32':
        data = r'E:\Dataset\kitti\yolo_fusion\data.yaml'
        device = 'cuda'
    elif sys.platform == 'darwin':
        data = r'/Users/harrier/Work/kitti/yolo_fusion/data.yaml'
        device = 'mps'
    
    model = FusionNet('ultralytics/cfg/models/v8/yolov8m-fusion.yaml', verbose=True)
    torchinfo.summary(model)
    # res = model.train(data=data, device=device, epochs=100, batch=2, cache='disk', workers= 4)

    # model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml', verbose=True)
    # torchinfo.summary(model)
    # res = model.train(data=data, device=device, epochs=100, batch=14, cache=True)

    
    # ln = LiDAR_norm()
    # lb = LetterBox_LiDAR()
    # fl = FusionLinear(4,345)
    # labels = {}

    # nploaded = np.load(r'E:\Dataset\kitti\yolo_fusion\train\images\000000.npz')
    # im, df = nploaded['im'], nploaded['df']
    
    # labels['img'] = im
    # labels['df'] = df

    # labels = ln(labels)
    # im1, df1 = lb(image=labels['img'], df=labels['df'])

    # nploaded = np.load(r'E:\Dataset\kitti\yolo_fusion\train\images\000001.npz')
    # im, df = nploaded['im'], nploaded['df']
    # labels['img'] = im
    # labels['df'] = df
    # labels = ln(labels)
    # im2, df2 = lb(image=labels['img'], df=labels['df'])

    # df = torch.stack((df1, df2))

    # y = fl(df)
    # print(y)

    # batch = torch.ones((3,12,12,6))
    # imgs = torch.split(batch, 3, -1)
    # print(imgs.shape)
