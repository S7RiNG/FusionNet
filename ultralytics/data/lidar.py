from pathlib import Path
import random
from PIL import Image
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import functional as f
import cv2
from .augment import LetterBox, Mosaic

def read_lidarmap(path) -> Tensor:
    return cv2.imread(str(path))

def read_lidarpoint(path) -> Tensor:
    return np.load(path)

def read_combo(img_path) -> Tensor:
    path = Path(img_path)
    path_map = path.parent/'..'/'maps'/path.name
    path_point = Path.with_suffix((path.parent/'..'/'points'/path.name), '.npy') 

    im = cv2.imread(str(path))
    # map = read_lidarmap(path_map)
    point = read_lidarpoint(path_point)
    # cat = np.concatenate([im, map], 2)
    return im, point

class LiDAR_norm:
    def __call__(self, labels=None, image=None, df=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        df = labels.get("df") if df is None else df

        df = df.astype(np.float32)

        h, w = labels.get("ori_shape")
        df[:,0] = df[:,0]/w
        df[:,1] = df[:,1]/h
        df[:,2:4] = df[:,2:4]/255

        if len(labels):
            labels["df"] = df
            return labels
        else:
            return df


class LetterBox_LiDAR(LetterBox):
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32, mode:bool='train'):
        super().__init__(new_shape, auto, scaleFill, scaleup, center, stride)
        self.mode = mode

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
        if shape[::-1] != new_unpad:  # resize
            rgb = cv2.resize(rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        if img.shape[-1] == 6:
            lid = img[:,:,3:]
            lid = cv2.resize(lid, new_unpad, interpolation=cv2.INTER_NEAREST)

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        rgb = cv2.copyMakeBorder(
            rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border to rgb

        if img.shape[-1] == 6:
            lid = cv2.copyMakeBorder(
                lid, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )  # add border to lidar
            img = np.concatenate((rgb, lid), axis=2)
        else:
            img = rgb

        #LiDAR point
        df = np.array(df)
        lidar_scale = np.array([new_unpad[0]/(new_unpad[0] + left + right), new_unpad[1]/(new_unpad[1] + top + bottom)], dtype=np.float32) #w, h scale
        lidar_offset = (1.0 - lidar_scale) * [
            1 if (left + right) == 0 else (left / (left + right)),
            1 if (top + bottom) == 0 else (top / (top + bottom))
            ]
        df[:, 0:2] = (df[:, 0:2] * lidar_scale)  + lidar_offset

        # df = df[np.argsort(df[:,2], kind="stable")[::-1]]
        np.random.shuffle(df)
        
        len_max = 28000
        len_df = df.shape[0]
        len_zero = len_max - len_df

        
        if len_zero > 0:
            zeros = np.zeros([len_zero, 4], dtype=df.dtype)
            df = np.concatenate([df[:len_df], zeros], 0)
        else:
            print('!!! LiDAR points exceed', len_max)
            df = df[:len_max]
        
        if self.mode == "train":
            noise = np.random.normal(0, 0.005, df.shape)
            df += noise
        
        df = df.T
        df = torch.from_numpy(df)
        
        if False:
            img_sh, lid = torch.split(torch.from_numpy(img), 3, -1)
            from matplotlib import pyplot as PLT
            pt_show = df
            shape_show = int(new_shape[1])
            pt_show[0:2] = pt_show[0:2] * shape_show
            u,v,z,i = pt_show
            PLT.figure(figsize=(12,5),dpi=96,tight_layout=True)
            PLT.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2) #'rainbow_r'
            PLT.axis([0,shape_show,shape_show,0])
            PLT.imshow(img_sh)
            PLT.show()


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
        
class LiDARAug():
    def __init__(self, mode='train', len_max=28000, randomchoice=0.8):
        self.mode = mode

    def __call__(self, labels=None, df=None, ):
        if labels is None:
            labels = {}
        df = labels.get("df") if df is None else df

        np.random.shuffle(df)
        
        len_max = 28000
        len_df = df.shape[0]
        len_zero = len_max - len_df

        
        if len_zero > 0:
            zeros = np.zeros([len_zero, 4], dtype=df.dtype)
            df = np.concatenate([df[:len_df], zeros], 0)
        else:
            print('!!! LiDAR points exceed', len_max)
            df = df[:len_max]
        
        if self.mode == "train":
            noise = np.random.normal(0, 0.005, df.shape)
            df += noise
        
        df = df.T
        df = torch.from_numpy(df)

class Mosic_Lidar(Mosaic):
    def _mosaic4(self, labels):
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            df = labels_patch["df"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                df4 = []
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            df_ymin = y1b/h
            df_ymax = y2b/h
            df_xmin = x1b/w
            df_xmax = x2b/w

            df_xscale = ((x2a - x1a) / (s * 2)) / ((x2b - x1b) / w)
            df_yscale = ((y2a - y1a) / (s * 2)) / ((y2b - y1b) / w)

            df_xoffset = x1a - x2a * df_xscale
            df_yoffset = y1a - y2a * df_yscale

            df_append = df[np.where(df[0,:] > df_xmin and df[0,:] < df_xmax and df[1,:] > df_ymin and df[1,:] < df_ymax)]
            df_append[0, :] = df_append[0, :] * df_xscale + df_xoffset
            df_append[1, :] = df_append[1, :] * df_yscale + df_yoffset

            df4.append(df_append)

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels