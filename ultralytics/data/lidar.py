from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import functional as f
import cv2
from .augment import LetterBox

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

class LiDAR_norm:
    def __call__(self, labels=None, image=None, df=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        df = labels.get("df") if df is None else df

        df = df.astype(np.float32)

        w, h = img.shape[:2]
        df[:,0] = df[:,0]/w
        df[:,1] = df[:,1]/h
        df[:,2:4] = df[:,2:4]/255

        if len(labels):
            labels["df"] = df
        else:
            return df


class LetterBox_LiDAR(LetterBox):
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        super().__init__(new_shape, auto, scaleFill, scaleup, center, stride)

    def __call__(self, labels=None, image=None, df=None):
        """
        Resizes and pads an image for object detection, instance segmentation, or pose estimation tasks.

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its
        aspect ratio and adding padding to fit the new shape. It also updates any associated labels accordingly.

        Args:
            labels (Dict | None): A dictionary containing image data and associated labels, or empty dict if None.
            image (np.ndarray | None): The input image as a numpy array. If None, the image is taken from 'labels'.

        Returns:
            (Dict | Tuple): If 'labels' is provided, returns an updated dictionary with the resized and padded image,
                updated labels, and additional metadata. If 'labels' is empty, returns a tuple containing the resized
                and padded image, and a tuple of (ratio, (left_pad, top_pad)).

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={"img": np.zeros((480, 640, 3)), "instances": Instances(...)})
            >>> resized_img = result["img"]
            >>> updated_instances = result["instances"]
        """
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        df = labels.get("df") if df is None else df
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        rgb = img[:,:,:3]
        lid = img[:,:,3:]
        if shape[::-1] != new_unpad:  # resize
            rgb = cv2.resize(rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
            lid = cv2.resize(lid, new_unpad, interpolation=cv2.INTER_NEAREST)
            

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        rgb = cv2.copyMakeBorder(
            rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border to rgb
        lid = cv2.copyMakeBorder(
            lid, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )  # add border to lidar

        img = np.concatenate((rgb, lid), axis=2)

        #LiDAR point
        lidar_scale = np.array(new_unpad, dtype=np.float32)/np.array(new_shape, dtype=np.float32)
        lidar_offset = (1 - lidar_scale)/2
        df[:, 0:2] = df[:, 0:2] * lidar_scale + lidar_offset


        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["df"] = df
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img, df