from ultralytics.engine.model import Model
from ultralytics.nn.tasks import FusionNetModel

from .predict import FusionNetPredictor
from .train import FusionNetTrainer
from .val import FusionNetValidator

class FusionNet(Model):
    def __init__(self, model = "yolov8m-fusion.yaml", verbose = False):
        super().__init__(model, 'detect', verbose)

    @property
    def task_map(self) -> dict:
        return {
            "detect": {
                "predictor": FusionNetPredictor,
                "validator": FusionNetValidator,
                "trainer": FusionNetTrainer,
                "model": FusionNetModel,
            }
        }
    
    def val(
        self,
        validator=None,
        **kwargs,
    ):
        """
        Validates the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for customization through various settings. It
        supports validation with a custom validator or the default validation approach. The method combines default
        configurations, method-specific defaults, and user-provided arguments to configure the validation process.

        Args:
            validator (ultralytics.engine.validator.BaseValidator | None): An instance of a custom validator class for
                validating the model.
            **kwargs (Any): Arbitrary keyword arguments for customizing the validation process.

        Returns:
            (ultralytics.utils.metrics.DetMetrics): Validation metrics obtained from the validation process.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # Print mAP50-95
        """
        args = {**self.overrides, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics