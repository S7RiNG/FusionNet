import sys, os, time
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

    epochs = 150
    batch = 2
    # time.sleep(3600)
    
    # Train
    # model = FusionNet('ultralytics/cfg/models/fusion/yolov8m-fusion.yaml', verbose=True)
    # model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml', verbose=True)
    # model = FusionNet(r'E:\Work\stu\FusionNet\ultralytics\cfg\models\fusion\yolov8m-fusion_5.yaml', verbose=True)
    model = FusionNet('/Users/harrier/Work/stu/FusionNet/ultralytics/cfg/models/fusion/yolov8s-fusion_5.yaml', verbose=True)
    # model = FusionNet(r'E:\Work\stu\FusionNet\runs\detect\fusion_aug1\weights\best.pt', verbose=True)
    # torchinfo.summary(model)

    res = model.train(data=data, device=device, epochs=epochs, batch=batch, cache='disk', workers=2, close_mosaic=20)

    # res = model.val(data=data, device=device, batch=batch, cache='disk', workers=4)

    # # Resume
    # model = FusionNet(r'E:\Work\stu\FusionNet\runs\detect\train4\weights\last.pt')
    # res = model.train(resume=True)

    # Val
    # model = FusionNet(r'E:\Work\stu\FusionNet\runs\detect\train\weights\best.pt')
    # torchinfo.summary(model)
    # res = model.val(data=data, device=device, batch=2, cache='disk')

    # model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml', verbose=True)
    # torchinfo.summary(model)
    # res = model.train(data=data, device=device, epochs=100, batch=32, cache='disk', augment=False)


    # ln = LiDAR_norm()
    # lb = LetterBox_LiDAR()
    # labels = {}

    # nploaded = np.load(r'E:\Dataset\kitti\yolo_fusion\train\images\000032.npz')
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