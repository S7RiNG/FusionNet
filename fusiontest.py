import sys
from ultralytics.models.fusion import FusionNet
from ultralytics.models.yolo import YOLO
import torchinfo 
from torch import nn
from ultralytics.data.lidar import LetterBox_LiDAR, LiDAR_norm
import numpy as np

if __name__ == "__main__":

    # if sys.platform == 'win':
    #     data = r'E:\Dataset\kitti\yolo_fusion\data.yaml'
    #     devoce = 'cuda'
    # elif sys.platform == 'darwin':
    #     data = r'/Users/harrier/Work/kitti/yolo_fusion/data.yaml'
    #     device = 'mps'
    
    # model = FusionNet('ultralytics/cfg/models/v8/yolov8m-fusion.yaml', verbose=True)
    # torchinfo.summary(model)
    # res = model.train(data=data, device=device, epochs=100, batch=14, cache='disk')

    # model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml', verbose=True)
    # torchinfo.summary(model)
    # res = model.train(data=data, device=device, epochs=100, batch=14, cache=True)

    lbox = LetterBox_LiDAR()
    lidar = LiDAR_norm()

    img = np.ones((300, 1000, 6), np.uint8)

    point = np.array([[100, 300, 124, 231], [150, 500, 231, 124]])

    df = lidar(image=img, df=point)

    print(df)

    imr, dfr = lbox(image=img, df=df)
    print(imr.shape, dfr)