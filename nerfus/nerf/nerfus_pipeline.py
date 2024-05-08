from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.engine.callbacks import TrainingCallbackLocation
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.writer import put_image
from typing_extensions import Literal


@dataclass
class NerfusPipelineConfig(VanillaPipelineConfig):
    """Nerfus Pipeline Config"""

    _target: Type = field(default_factory=lambda: NerfusPipeline)

    probe_depth: float = 0.14
    probe_width: float = 0.08

    use_border_loss: bool = True
    border_loss_mult: float = 1e-1
    use_scatter_density_loss: bool = True
    scatter_density_loss_mult: float = 1e-1

    num_cubes: int = 40
    cube_resolution: int = 32
    cube_scale_perc_range: Tuple[float, float] = (0.01, 0.10)
    cube_start_step: int = 0
    steps_per_draw_cubes: int = 20
    singlestep_starting_t: int = 10

    visualize_cubes: bool = True
    steps_per_visualize_cubes: int = 100
    max_num_cubes_to_visualize: int = 6

    kernel_size: int = 3
    kernel_std: float = 1.0


class NerfusPipeline(VanillaPipeline):
    config: NerfusPipelineConfig

    def __init__(
        self,
        config: NerfusPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        kernel = self._create_gaussian_kernel()
        self.register_buffer("gaussian_kernel", kernel)

    def _create_gaussian_kernel(self):
        size = self.config.kernel_size
        std = self.config.kernel_std

        x = torch.arange(-size, size + 1, dtype=torch.float32, device=self.device)
        gaussian1D_x = torch.exp(-0.5 * ((x) / (std * 2)) ** 2)
        gaussian1D_y = torch.exp(-0.5 * ((x) / std) ** 2)

        kernel_2d = torch.outer(gaussian1D_x, gaussian1D_y)
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d[None, None, :, :]

    def render_ultrasound(
        self, encoded_positions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        features = encoded_positions.mean(dim=-1)
        return {
            "intensity": torch.sigmoid(features),
            "border_prob": torch.sigmoid(features + 0.1),
            "scatter_density": torch.sigmoid(features - 0.1),
            "raw_params": torch.tanh(encoded_positions[..., :3]),
        }

    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)

        ones = torch.ones_like(ray_bundle.origins[..., 0:1])
        ray_bundle.nears = ones * 0.0
        ray_bundle.fars = ones * self.config.probe_depth

        self.model.field = self.model.field.to(self.device)
        self.model.field.backbone = self.model.field.backbone.to(self.device)

        model_outputs = self.model(ray_bundle)
        encoded_positions = model_outputs["encoded_positions"].to(self.device)

        features = self.model.field(encoded_positions)

        intensity = torch.sigmoid(features[..., 0])
        border_prob = torch.sigmoid(features[..., 1])
        scatter_density = torch.sigmoid(features[..., 2])

        image = batch["image"].to(self.device)
        if image.shape[-1] == 3:
            image = (
                0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
            )

        ultrasound_outputs = {
            "intensity": intensity,
            "border_prob": border_prob,
            "scatter_density": scatter_density,
            "raw_params": features[..., :3],
        }
        model_outputs.update(ultrasound_outputs)

        intensity = intensity.reshape(image.shape)

        recon_loss = F.mse_loss(intensity, image)
        loss_dict = {"recon_loss": recon_loss}

        if step >= self.config.cube_start_step and self.config.use_border_loss:
            denoised_border = self.model.get_denoised_border_prob(encoded_positions)
            border_loss = F.mse_loss(border_prob, denoised_border)
            loss_dict["border_loss"] = self.config.border_loss_mult * border_loss

        if step >= self.config.cube_start_step and self.config.use_scatter_density_loss:
            denoised_scatter = self.model.get_denoised_scatter_density(
                encoded_positions
            )
            scatter_loss = F.mse_loss(scatter_density, denoised_scatter)
            loss_dict["scatter_loss"] = (
                self.config.scatter_density_loss_mult * scatter_loss
            )

        metrics_dict = {
            "psnr": -10 * torch.log10(recon_loss),
            "pred_intensity_mean": intensity.mean().detach(),
            "pred_border_mean": border_prob.mean().detach(),
            "pred_scatter_mean": scatter_density.mean().detach(),
        }

        return model_outputs, loss_dict, metrics_dict

    def get_eval_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_eval(step)

        ones = torch.ones_like(ray_bundle.origins[..., 0:1])
        ray_bundle.nears = ones * 0.0
        ray_bundle.fars = ones * self.config.probe_depth

        model_outputs = self.model(ray_bundle)
        raw_outputs = model_outputs["raw_outputs"]

        ultrasound_outputs = self.render_ultrasound(raw_outputs)
        model_outputs.update(ultrasound_outputs)

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        loss = F.mse_loss(ultrasound_outputs["intensity"], batch["image"])
        loss_dict = {"val_loss": loss}

        metrics_dict["val_psnr"] = -10 * torch.log10(loss)

        return model_outputs, loss_dict, metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ):
        metrics = {}
        images = {}

        mse = F.mse_loss(outputs["intensity"], batch["image"])
        metrics["psnr"] = -10 * torch.log10(mse)

        images["intensity"] = outputs["intensity"]
        images["gt"] = batch["image"]
        images["border_prob"] = outputs["border_prob"]
        images["scatter_density"] = outputs["scatter_density"]

        return metrics, images

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []

        callbacks.extend(
            training_callback_attributes.pipeline.datamanager.get_training_callbacks(
                training_callback_attributes
            )
        )
        callbacks.extend(
            training_callback_attributes.pipeline.model.get_training_callbacks(
                training_callback_attributes
            )
        )

        if training_callback_attributes.viewer_state and self.config.visualize_cubes:

            def visualize_cubes(step):
                if step % self.config.steps_per_visualize_cubes == 0:
                    outputs = self._get_latest_outputs()
                    if outputs is not None:
                        writer.put_image(
                            name="border_prob", image=outputs["border_prob"], step=step
                        )
                        writer.put_image(
                            name="scatter_density",
                            image=outputs["scatter_density"],
                            step=step,
                        )
                        writer.put_image(
                            name="intensity", image=outputs["intensity"], step=step
                        )

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=self.config.steps_per_visualize_cubes,
                    func=visualize_cubes,
                )
            )

        return callbacks

    def _get_latest_outputs(self):
        try:
            ray_bundle, batch = self.datamanager.next_train(0)
            model_outputs = self.model(ray_bundle)
            ultrasound_outputs = self.render_ultrasound(model_outputs["raw_outputs"])
            return ultrasound_outputs
        except Exception as e:
            print(f"Error getting latest outputs: {e}")
            return None
