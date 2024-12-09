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