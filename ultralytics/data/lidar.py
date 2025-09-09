from pathlib import Path
import random
from PIL import Image
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import functional as f
import cv2
import math

from ultralytics.utils import LOGGER
from ultralytics.utils.instance import Instances
from ultralytics.data.augment import Compose, CopyPaste, LetterBox, Mosaic, RandomFlip, RandomHSV, RandomPerspective, Albumentations

def read_lidarmap(path) -> Tensor:
    map = np.array(Image.open(str(path)))
    if len(map.shape) < 3:
        return np.expand_dims(map, -1)
    else:
        return map
    # return cv2.imread(str(path))

def read_lidarpoint(path) -> Tensor:
    return np.load(path)

def read_combo(img_path) -> Tensor:
    path = Path(img_path)
    path_map = path.parent/'..'/'maps'/path.name
    path_map9ch = Path.with_suffix((path.parent/'..'/'map9ch'/path.name), '.npy') 
    path_map4ch = Path.with_suffix((path.parent/'..'/'map4ch'/path.name), '.npy') 
    path_depth = path.parent/'..'/'depths'/path.name
    path_point = Path.with_suffix((path.parent/'..'/'points'/path.name), '.npy') 

    im = cv2.imread(str(path))

    if False: #points
        df = np.load(path_point)
    elif False:
        df = read_lidarmap(path_map)
    elif False: #path_map9ch
        df = np.load(path_map9ch)
    elif True: #path_map4ch
        df = np.load(path_map4ch)
    else:
        df = read_lidarmap(path_depth)
    return im, df

class LiDAR_norm:
    def __call__(self, labels=None, image=None, df=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        df = labels.get("df") if df is None else df
        df = df.astype(np.float32)
        if len(df.shape) == 3:
            labels["df"] = df
            return labels

        h, w = labels.get("ori_shape")
        df[:,0] = df[:,0]/w
        df[:,1] = df[:,1]/h
        df[:,2:4] = df[:,2:4]/255
        df = torch.from_numpy(df)

        if len(labels):
            labels["df"] = df
            return labels
        else:
            return df

# class Process_LiDAR:
#     def __init__(self, lenmax=28000, mode:str='train'):
#         self.lenmax = lenmax
#         self.mode = mode

#     def __call__(self, labels=None, df=None):
#         df = labels.get("df") if df is None else df
#         df = df.T

#         #add noise
#         if self.mode == "train":
#             noise = np.random.normal(0, 0.001, df.shape)
#             df += noise
        
#         #shuffle
#         np.random.shuffle(df)

#         #limit length
#         len_max = self.lenmax
#         len_df = df.shape[0]
#         len_zero = len_max - len_df
        
#         if len_zero <= 0:
#             if self.mode == "val":
#                 print('!!! LiDAR points exceed', len_max)
#                 df = df[:len_max]
#             len_zero = np.random.randint(1000, 2000)
#             len_df = len_max - len_zero
        
#         zeros = np.zeros([len_zero, 4], dtype=df.dtype)
#         df = np.concatenate([df[:len_df], zeros], 0)
        
#         df = df.T
#         if False:
#             from matplotlib import pyplot as PLT
#             pt_show = df
#             shape_show = 640
#             pt_show[0:2] = pt_show[0:2] * shape_show
#             u,v,z,i = pt_show
#             PLT.figure(figsize=(12,5),dpi=96,tight_layout=True)
#             PLT.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2) #'rainbow_r'
#             PLT.axis([0,shape_show,shape_show,0])
#             PLT.imshow(labels['img'])
#             PLT.show()
#             while True: continue
#         if labels is None:
#             return df
#         else:
#             labels["df"] = df
#             return labels

class LetterBox_LiDAR(LetterBox):
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32, mode:str='train'):
        super().__init__(new_shape, auto, scaleFill, scaleup, center, stride)
        self.mode = mode if mode=='train' else 'val'

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

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        if len(df.shape) == 3:
            df = cv2.resize(df, new_unpad, interpolation=cv2.INTER_LINEAR)
            if len(df.shape) != 3:
                df = np.expand_dims(df, -1)

        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border to rgb

        if len(df.shape) == 3:
            # df = np.pad(df, pad_width=((top, bottom), (left, right), (0,0)), mode="constant", constant_values=114)
            # df = cv2.copyMakeBorder(
            #     df, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114
            # )  # add border to lidar

            dim = df.shape[-1]
            buf = []
            for slic_dim in range(0, dim, 3):
                slic_dim_end = min(slic_dim+3, dim)
                slic = df[:,:,slic_dim:slic_dim_end]

                ndim = slic_dim_end - slic_dim_end
                value = [114,] * ndim
                slic = cv2.copyMakeBorder(
                    slic, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value
                )  # add border to lidar
                if len(slic.shape) != 3:
                    slic = np.expand_dims(slic, -1)
                buf.append(slic)
            df = np.concatenate(buf, -1)
           
            if len(df.shape) != 3:
                df = np.expand_dims(df, -1)
        else:
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
            df = df.T
            # df = torch.from_numpy(df)
        
        # img_sh, lid = torch.split(torch.from_numpy(img), 3, -1)
        # from matplotlib import pyplot as PLT
        # pt_show = df
        # shape_show = int(new_shape[1])
        # pt_show[0:2] = pt_show[0:2] * shape_show
        # u,v,z,i = pt_show
        # PLT.figure(figsize=(12,5),dpi=96,tight_layout=True)
        # PLT.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2) #'rainbow_r'
        # PLT.axis([0,shape_show,shape_show,0])
        # PLT.imshow(img_sh)
        # PLT.show()


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

class Mosic_LiDAR(Mosaic):
    def __init__(self, dataset, imgsz=640, p=1.0, n=4, pre_transform=None):
        """
        Initializes the Mosaic augmentation object.

        This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
        The augmentation is applied to a dataset with a given probability.

        Args:
            dataset (Any): The dataset on which the mosaic augmentation is applied.
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
            n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).

        Examples:
            >>> from ultralytics.data.augment import Mosaic
            >>> dataset = YourDataset(...)
            >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        """
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        super(Mosaic, self).__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n

    def _mosaic4(self, labels):
        if len(labels["df"].shape) == 3:
            imdf = True
        else:
            imdf = False
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            df = labels_patch["df"]

            if imdf:
                img = np.concatenate((img, df), axis=2)

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

            if imdf == False:
                df_ymin = y1b/h
                df_ymax = y2b/h
                df_xmin = x1b/w
                df_xmax = x2b/w

                dft = df.T
                lim1 = dft[0] > df_xmin
                lim2 = dft[0] < df_xmax
                lim3 = dft[1] > df_ymin
                lim4 = dft[1] < df_ymax
                limx = np.logical_and(lim1, lim2)
                limy = np.logical_and(lim3, lim4)
                lim = np.logical_and(limx, limy)

                df_xscale = ((x2a - x1a) / (s * 2)) / ((x2b - x1b) / w)
                df_yscale = ((y2a - y1a) / (s * 2)) / ((y2b - y1b) / h)

                df_xoffset = (x1a / (s * 2) - (x1b / w) * df_xscale)
                df_yoffset = (y1a / (s * 2) - (y1b / w) * df_yscale)

                df_append = df[np.where(lim)]
                df_append[:, 0] = df_append[:, 0] * df_xscale + df_xoffset
                df_append[:, 1] = df_append[:, 1] * df_yscale + df_yoffset

                df4.append(df_append)
            
            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        
        if imdf:
            df4 = img4[:,:,3:]
            img4 = img4[:,:,:3]
        else:
            df4 = np.concatenate(df4, axis=0).T
            
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        final_labels["df"] = df4

        if False:
            from matplotlib import pyplot as PLT
            pt_show = df4
            shape_show = int(s * 2)
            pt_show[0:2] = pt_show[0:2] * shape_show
            u,v,z,i = pt_show
            PLT.figure(figsize=(12,5),dpi=96,tight_layout=True)
            PLT.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2) #'rainbow_r'
            PLT.axis([0,shape_show + 100,shape_show + 100,0])
            PLT.imshow(img4)
            PLT.show()

        return final_labels
    
class RandomPerspective_LiDAR(RandomPerspective):
    def __call__(self, labels):
        """
        Applies random perspective and affine transformations to an image and its associated labels.

        This method performs a series of transformations including rotation, translation, scaling, shearing,
        and perspective distortion on the input image and adjusts the corresponding bounding boxes, segments,
        and keypoints accordingly.

        Args:
            labels (Dict): A dictionary containing image data and annotations.
                Must include:
                    'img' (ndarray): The input image.
                    'cls' (ndarray): Class labels.
                    'instances' (Instances): Object instances with bounding boxes, segments, and keypoints.
                May include:
                    'mosaic_border' (Tuple[int, int]): Border size for mosaic augmentation.

        Returns:
            (Dict): Transformed labels dictionary containing:
                - 'img' (np.ndarray): The transformed image.
                - 'cls' (np.ndarray): Updated class labels.
                - 'instances' (Instances): Updated object instances.
                - 'resized_shape' (Tuple[int, int]): New image shape after transformation.

        Examples:
            >>> transform = RandomPerspective()
            >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> labels = {
            ...     "img": image,
            ...     "cls": np.array([0, 1, 2]),
            ...     "instances": Instances(bboxes=np.array([[10, 10, 50, 50], [100, 100, 150, 150]])),
            ... }
            >>> result = transform(labels)
            >>> assert result["img"].shape[:2] == result["resized_shape"]
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # do not need ratio pad

        img = labels["img"]
        cls = labels["cls"]
        df = labels["df"]
        shape_in = img.shape[:2][::-1]
        instances = labels.pop("instances")
        # Make sure the coord formats are right
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*shape_in)

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # Scale for func:`box_candidates`
        img, M, scale, df = self.affine_transform(img, border, df, shape_in)
        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        # Clip
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        labels["df"] = df
        if False:
            from matplotlib import pyplot as PLT
            pt_show = df
            shape_show = 640
            pt_show[0:2] = pt_show[0:2] * shape_show
            u,v,z,i = pt_show
            PLT.figure(figsize=(12,5),dpi=96,tight_layout=True)
            PLT.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2) #'rainbow_r'
            PLT.axis([0, shape_show, shape_show,0])
            PLT.imshow(img)
            PLT.show()
            while True: pass
        return labels
    
    def affine_transform(self, img, border, df, shape_in):
        """
        Applies a sequence of affine transformations centered around the image center.

        This function performs a series of geometric transformations on the input image, including
        translation, perspective change, rotation, scaling, and shearing. The transformations are
        applied in a specific order to maintain consistency.

        Args:
            img (np.ndarray): Input image to be transformed.
            border (Tuple[int, int]): Border dimensions for the transformed image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, float]): A tuple containing:
                - np.ndarray: Transformed image.
                - np.ndarray: 3x3 transformation matrix.
                - float: Scale factor applied during the transformation.

        Examples:
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> border = (10, 10)
            >>> transformed_img, matrix, scale = affine_transform(img, border)
        """
        # Center
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))



            if len(df.shape) == 3:
                dim = df.shape[-1]
                buf = []
                for slic_dim in range(0, dim, 3):
                    slic_dim_end = min(slic_dim+3, dim)
                    slic = df[:,:,slic_dim:slic_dim_end]

                    ndim = slic.shape[-1]
                    value = [114, ] * ndim
                    if self.perspective:
                        slic = cv2.warpPerspective(slic, M, dsize=self.size, borderValue=value)
                    else:
                        slic = cv2.warpAffine(slic, M[:2], dsize=self.size, borderValue=value)
                    
                    if len(slic.shape) != 3:
                        slic = np.expand_dims(slic, -1)
                    buf.append(slic)
                df = np.concatenate(buf, -1)
            if len(df.shape) != 3:
                df = np.expand_dims(df, -1)
        if len(df.shape) != 3:
            df = self.apply_lidar(M, df, shape_in, img.shape[:2][::-1])
        return img, M, s, df
    
    def apply_lidar(self, M:np.matrix, df:np.matrix, shape, shape_out, border):
        if len(df.shape) == 3:
            df, M, scale = self.affine_transform(df, border)
            if len(df.shape) != 3:
                df = np.expand_dims(df, -1)
            return df
        else:
            # denorm
            denorm = np.array(shape).reshape([-1, 2])
            norm = np.array(shape_out).reshape([-1, 2])
            dft = df.T # n first
            xy = np.ones([dft.shape[0], 3])
            xy[:, :2] = dft[:, :2] * denorm
            xy = xy @ M.T
            xy[:, :2] = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2])
            xy[:, :2] = xy[:, :2] / norm

            dft_affine = np.empty(np.shape(dft))
            dft_affine[:, :2] = xy[:, :2]
            dft_affine[:, 2:] = dft[:, 2:]

            lim1 = dft_affine.T[0] < 1
            lim2 = dft_affine.T[0] > 0
            lim3 = dft_affine.T[1] < 1
            lim4 = dft_affine.T[1] > 0
            limx = np.logical_and(lim1, lim2)
            limy = np.logical_and(lim3, lim4)
            lim = np.logical_and(limx, limy)

            dft_affine = dft_affine[lim]
            return dft_affine.T
        
class RandomFlip_LiDAR(RandomFlip):
    def __call__(self, labels):
        """
        Applies random flip to an image and updates any instances like bounding boxes or keypoints accordingly.

        This method randomly flips the input image either horizontally or vertically based on the initialized
        probability and direction. It also updates the corresponding instances (bounding boxes, keypoints) to
        match the flipped image.

        Args:
            labels (Dict): A dictionary containing the following keys:
                'img' (numpy.ndarray): The image to be flipped.
                'instances' (ultralytics.utils.instance.Instances): An object containing bounding boxes and
                    optionally keypoints.

        Returns:
            (Dict): The same dictionary with the flipped image and updated instances:
                'img' (numpy.ndarray): The flipped image.
                'instances' (ultralytics.utils.instance.Instances): Updated instances matching the flipped image.

        Examples:
            >>> labels = {"img": np.random.rand(640, 640, 3), "instances": Instances(...)}
            >>> random_flip = RandomFlip(p=0.5, direction="horizontal")
            >>> flipped_labels = random_flip(labels)
        """
        img = labels["img"]
        df = labels["df"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        # Flip up-down
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
            if len(df.shape) == 3:
                df = np.flipud(df)
            else:
                df[1] = 1 - df[1]
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            if len(df.shape) == 3:
                df = np.fliplr(df)
            else:
                df[0] = 1 - df[0]
            # For keypoints
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        labels["df"] = df

        return labels

class Grouping_LiDAR:
    def __call__(self, labels:np.ndarray):
        img = labels["img"]
        df = labels["df"] # c, n

        if len(df.shape) == 3: #norm
            df = df / 255
            labels["df"] = torch.from_numpy(df).permute(2,0,1)
            return labels
        
        depth = 8
        
        data = df.T
        data[:,2] = 1 - data[:,2]

        shape = img.shape[:2]
        w, h = shape

        
        group = np.zeros((w, h, depth), dtype=np.float32)
        groupidx = np.zeros((w, h, 1), dtype=np.int8)
        scale = np.ones(data.shape[-1])
        scale[0] = w
        scale[1] = h
        np.asarray(scale, dtype=np.float32)
        data = data * scale

        for pt in data:
            x, y = int(pt[0]), int(pt[1])

            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            
            idx = groupidx[x][y]
            if idx >= depth:
                continue

            if group[x][y][idx] == 0.0:
                group[x][y][idx] = pt[2]
                groupidx[x][y] += 1

        groupidx = np.asarray(groupidx, dtype=group.dtype) / depth
        group = np.concatenate((groupidx, group), 2)

        labels["df"] = df = torch.from_numpy(group.transpose(2,0,1)) #c, w, h
        return labels
    
# class Grouping_LiDAR:
#     def __call__(self, labels:np.ndarray):
#         img = labels["img"]
#         df = labels["df"] # c, n
#         depth = 4
        
#         data = df.T
#         data[:,3] = 1
#         shape = img.shape[:2]
#         w, h = shape

#         groupsize = (w, h, depth, data.shape[-1])
#         group = np.zeros(groupsize)
#         scale = np.ones(data.shape[-1])
#         scale[0] = w
#         scale[1] = h
#         np.asarray(scale, dtype=np.float32)
#         data = data * scale
#         zero_data = np.zeros(data.shape[-1])

#         for pt in data:
#             x, y = int(pt[0]), int(pt[1])

#             if x < 0 or x >= w or y < 0 or y >= h:
#                 continue

#             for d in range(depth):
#                 if np.array_equal(group[x][y][d], zero_data):
#                     pt[0] -= x
#                     pt[1] -= y
#                     group[x][y][d] = pt
#                     break

#         # group[:,:,:,0] -= 0.5
#         # group[:,:,:,1] -= 0.5

#         labels["df"] = df = torch.from_numpy(group.reshape(w, h, -1).transpose(2,0,1)) #c, w, h
#         return labels

def LiDAR_transforms(dataset, imgsz, hyp, stretch=False):
    """
    Applies a series of image transformations for training.

    This function creates a composition of image augmentation techniques to prepare images for YOLO training.
    It includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (Namespace): A dictionary of hyperparameters controlling various aspects of the transformations.
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.

    Returns:
        (Compose): A composition of image transformations to be applied to the dataset.

    Examples:
        >>> from ultralytics.data.dataset import YOLODataset
        >>> from ultralytics.utils import IterableSimpleNamespace
        >>> dataset = YOLODataset(img_path="path/to/images", imgsz=640)
        >>> hyp = IterableSimpleNamespace(mosaic=1.0, copy_paste=0.5, degrees=10.0, translate=0.2, scale=0.9)
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])
    """
    mosaic = Mosic_LiDAR(dataset, imgsz=imgsz, p=hyp.mosaic, pre_transform=LiDAR_norm())
    affine = RandomPerspective_LiDAR(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox_LiDAR(new_shape=(imgsz, imgsz)),
    )

    pre_transform = Compose([LiDAR_norm(), mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([LiDAR_norm(), Mosic_LiDAR(dataset, imgsz=imgsz, p=hyp.mosaic, pre_transform=LiDAR_norm()), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            pre_transform,
            # MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip_LiDAR(direction="vertical", p=hyp.flipud),
            RandomFlip_LiDAR(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
            # Process_LiDAR(),
            Grouping_LiDAR(),
        ]
    )  # transforms