from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.fusion import FusionNetValidator
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.data.build import build_fusion_dataset

from copy import copy

class FusionNetTrainer(DetectionTrainer):
    def __init__(self, cfg=..., overrides=None, _callbacks=None):
        overrides['task'] = 'fusion'
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build FusionNet Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_fusion_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def preprocess_batch(self, batch):
        """Add data of fusion : 'dfs' """
        batch['dfs'] = batch['dfs'].to(self.device, non_blocking=True).float()
        return super().preprocess_batch(batch)
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        return super().get_model(cfg, weights, verbose)
    
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return FusionNetValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    
    
