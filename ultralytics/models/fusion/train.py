import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.data.build import build_fusion_dataset

from ultralytics.utils.plotting import plot_images

from .val import FusionNetValidator
from .model import FusionNetModel

from copy import copy

class FusionNetTrainer(DetectionTrainer):
    # def __init__(self, overrides=None):
    #     super().__init__(overrides)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build FusionNet Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_fusion_dataset(self.args, img_path, batch, self.data, mode=mode, stride=gs, rect=False)

    def preprocess_batch(self, batch):
        """Add data of fusion : 'dfs' """
        batch['df'] = batch['df'].float().to(self.device, non_blocking=True)
        return super().preprocess_batch(batch)
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = FusionNetModel(cfg, nc=self.data["nc"], verbose=verbose)
        if weights:
            model.load(weights)
        return model
    
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return FusionNetValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        if len(batch["df"][0].shape) == 3:
            dfs = batch["df"]
            dfs_show = []
            for df in dfs:
                if df.shape[0] < 3:
                    df = torch.cat((df, torch.zeros(3-df.shape[0], df.shape[1], df.shape[2], dtype=df.dtype, device=df.device)))
                elif df.shape[0] > 3:
                    df = df[-3:,:,:]
                dfs_show.append(df)
            dfs_show = torch.stack(dfs_show, dim=0)
            plot_images(
                images=dfs_show,
                batch_idx=batch["batch_idx"],
                cls=batch["cls"].squeeze(-1),
                bboxes=batch["bboxes"],
                paths=batch["im_file"],
                fname=self.save_dir / f"train_batch_LiDAR{ni}.jpg",
                on_plot=self.on_plot,
            )

        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch_RGB{ni}.jpg",
            on_plot=self.on_plot,
        )


    
