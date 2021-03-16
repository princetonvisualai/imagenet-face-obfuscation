import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Any


class GradCamWrapper:
    """
    Wrap a model to produce Grad-CAM.
    
    @inproceedings{selvaraju2017grad,
        title={Grad-cam: Visual explanations from deep networks via gradient-based localization},
        author={Selvaraju, Ramprasaath R and Cogswell, Michael and Das, Abhishek and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
        booktitle={International Conference on Computer Vision (ICCV)},
        year={2017}
    }
    """

    model_name: str  # Name of the model, e.g., AlexNet, ResNet50, etc.
    gradients: List[torch.Tensor]  # Gradients w.r.t. the spatial feature map

    def __init__(self, model: Any) -> None:
        name = model._get_name()
        if name in ["DataParallel", "DistributedDataParallel"]:
            model = model.module
            name = model._get_name()
        submodules = list(model.children())

        if name == "MobileNetV2":
            self.features = model.features
            self.classifier = nn.Sequential(
                nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), model.classifier  # type: ignore
            )
        elif name in ["AlexNet", "VGG"]:
            self.features = model.features
            self.classifier = nn.Sequential(
                model.avgpool, nn.Flatten(), model.classifier  # type: ignore
            )
        elif name == "DenseNet":
            self.features = model.features
            self.classifier = nn.Sequential(
                nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), model.classifier  # type: ignore
            )
        elif name == "SqueezeNet":
            self.features = model.features
            self.classifier = nn.Sequential(model.classifier, nn.Flatten())  # type: ignore
        elif name == "ResNet":
            self.features = nn.Sequential(*submodules[:-2])
            self.classifier = nn.Sequential(model.avgpool, nn.Flatten(), model.fc)  # type: ignore
        elif name == "ShuffleNetV2":
            self.features = nn.Sequential(*submodules[:-1])
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), model.fc  # type: ignore
            )
        else:
            raise NotImplementedError

        self.model_name = name
        self.gradients = []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.feature_maps = self.features(x)
        self.feature_maps.register_hook(lambda grad: self.gradients.append(grad))
        return self.classifier(self.feature_maps)

    def eval(self) -> None:
        self.features.eval()
        self.classifier.eval()

    def zero_grad(self) -> None:
        self.features.zero_grad()
        self.classifier.zero_grad()
        self.gradients = []

    def grad_cam(self, output: torch.Tensor, target: torch.Tensor) -> Tuple[Any, Any]:
        predicted_scores, _ = output.max(dim=-1)
        sum_predicted_scores = predicted_scores.sum()

        # cam for the ground truth class
        target_scores = (output * F.one_hot(target, num_classes=1000)).sum(dim=-1)
        sum_target_scores = target_scores.sum()

        self.zero_grad()
        sum_target_scores.backward(retain_graph=True)
        with torch.no_grad():
            weights = self.gradients[0].mean([2, 3])
            cam_gt = (self.feature_maps * weights.unsqueeze(-1).unsqueeze(-1)).sum(
                dim=1
            )
            cam_gt = F.relu(cam_gt)
            cam_gt -= cam_gt.min()
            cam_gt /= cam_gt.max()

        # cam for the predicted class
        self.zero_grad()
        sum_predicted_scores.backward(retain_graph=True)  # type: ignore
        with torch.no_grad():
            weights = self.gradients[0].mean([2, 3])
            cam_pred = (self.feature_maps * weights.unsqueeze(-1).unsqueeze(-1)).sum(
                dim=1
            )
            cam_pred = F.relu(cam_pred)
            cam_pred -= cam_pred.min()
            cam_pred /= cam_pred.max()

        return cam_gt.cpu().numpy(), cam_pred.cpu().numpy()
