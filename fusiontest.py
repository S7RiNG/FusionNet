from ultralytics.models.fusion import FusionNet
from ultralytics.models.yolo import YOLO
import torchinfo 
from torch import nn

if __name__ == "__main__":

    model = FusionNet('ultralytics/cfg/models/v8/yolov8m-fusion.yaml', verbose=True)
    torchinfo.summary(model)
    res = model.train(data=r'E:\Dataset\kitti\yolo_fusion\data.yaml',device='cuda', epochs=100, batch=14, cache='disk')

# model = YOLO('ultralytics/cfg/models/v8/yolov8m.yaml', verbose=True)
# torchinfo.summary(model)
# res = model.train(data=r'E:\Dataset\kitti\yolo_fusion\data.yaml',device='cuda', epochs=100, batch=14, cache=True)