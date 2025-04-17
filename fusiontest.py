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
        data = r'E:\Dataset\kitti\yolo_fusion_resize\data.yaml'
        device = 'cuda'
        modelyaml = r'E:\Work\stu\FusionNet\ultralytics\cfg\models\fusion\yolov8m-fusion_18.yaml'
    elif sys.platform == 'darwin':
        data = r'/Users/harrier/Work/kitti/yolo_fusion/data.yaml'
        device = 'mps'
        modelyaml = '/Users/harrier/Work/stu/FusionNet/ultralytics/cfg/models/fusion/yolov8m-fusion_20.yaml'

    if True:
        modelcreater = YOLO
    else:
        modelcreater = FusionNet

    epochs = 100
    batch = 8
    
    # time.sleep(3600)
    
    # Train
    # model = FusionNet('ultralytics/cfg/models/fusion/yolov8m-fusion.yaml', verbose=True)
    # model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml', verbose=True)
    model = FusionNet(modelyaml, verbose=True)
    # model = FusionNet(r'E:\Work\stu\FusionNet\runs\detect\train4\weights\best.pt', verbose=True)
    # # print(model)

    res = model.train(data=data, device=device, epochs=epochs, batch=batch, cache='disk', workers=7)

    # res = model.val(data=data, device=device, batch=batch, cache='disk', workers=7)

    # # Resume mig
    # model = FusionNet(r'E:\Work\stu\FusionNet\runs\detect\train8\weights\last.pt')
    # res = model.train(resume=True)
    # res = model.train(data=data, device=device, epochs=epochs, batch=batch, cache='disk', workers=6)

    # Val
    # model = FusionNet(r'E:\Work\stu\FusionNet\runs\detect\train\weights\best.pt')
    # torchinfo.summary(model)
    # res = model.val(data=data, device=device, batch=batch, cache='disk', workers=5)

    # model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml', verbose=True)
    # res = model.train(data=data, device=device, epochs=epochs, batch=batch, cache='disk')


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