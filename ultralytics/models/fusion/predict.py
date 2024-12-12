# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.data.lidar import LetterBox_LiDAR


class FusionNetPredictor(DetectionPredictor):
    def pre_transform(self, im):
        """
        Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Returns:
            (list): A list of transformed images.
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox_LiDAR(
            self.imgsz,
            auto=same_shapes and (self.model.pt or getattr(self.model, "dynamic", False)),
            stride=self.model.stride,
        )
        return [letterbox(image=x) for x in im]
