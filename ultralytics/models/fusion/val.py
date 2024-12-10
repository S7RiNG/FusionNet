from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.build import build_fusion_dataset
from ultralytics.utils.plotting import plot_images, output_to_target

class FusionNetValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

    def preprocess(self, batch):
        batch["df"] = batch["df"].to(self.device, non_blocking=True)
        super().preprocess(batch)
    

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_fusion_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)
    
    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        imgs = [img[:,:,:3] for img in  batch["img"]] #RGB
        plot_images(
            imgs,
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        imgs = [img[:,:,:3] for img in  batch["img"]] #RGB
        plot_images(
            imgs,
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred