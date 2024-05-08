import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dotmap import DotMap
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.encodings import NeRFEncoding as SinusoidalEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from nerfstudio.model_components.ray_samplers import UniformSampler
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.rich_utils import CONSOLE
from PIL import Image

from nerfus.lightning.nerfus_trainer import NerfusTrainer
from nerfus.nerf.nerfus_pipeline import NerfusPipelineConfig


class NerfusField(Field):
    def __init__(
        self,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        cube_resolution: int = 128,
        input_dim: int = 63,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.spatial_distortion = spatial_distortion
        self.cube_resolution = cube_resolution
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        backbone_layers = []
        curr_dim = input_dim

        for _ in range(hidden_layers):
            backbone_layers.extend([nn.Linear(curr_dim, hidden_dim), nn.ReLU(True)])
            curr_dim = hidden_dim

        backbone_layers.append(nn.Linear(curr_dim, 5))
        self.backbone = nn.Sequential(*backbone_layers)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        original_shape = positions.shape
        positions = positions.reshape(-1, self.input_dim)

        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)

        output = self.backbone(positions)
        return output.reshape(*original_shape[:-1], 5)

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        if ray_bundle is None:
            raise ValueError("Ray bundle cannot be None")

        origins = ray_bundle.origins
        directions = ray_bundle.directions
        starts = ray_bundle.nears
        ends = ray_bundle.fars

        mid = (starts + ends) / 2.0
        mid = mid.unsqueeze(-1)
        positions = origins + directions * mid

        encoded = self.position_encoding(positions)

        if encoded is None:
            raise ValueError("Position encoding returned None")

        return {"encoded_positions": encoded}

    def to(self, device):
        super().to(device)
        self.backbone = self.backbone.to(device)
        return self


@dataclass
class NerfusModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: NerfusModel)

    hidden_dim: int = 256
    hidden_layers: int = 2
    cube_resolution: int = 128
    use_spatial_distortion: bool = False
    cube_start_step: int = 0
    use_border_loss: bool = True
    border_loss_mult: float = 1.0
    use_scatter_density_loss: bool = True
    scatter_density_loss_mult: float = 1.0

    num_samples: int = 64

    diffusion_config_path: Optional[Path] = None
    diffusion_ckpt_path: Optional[Path] = None

    probe_depth: float = 10.0

    singlestep_starting_t: int = 10

    position_encoding_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "in_dim": 3,
            "num_frequencies": 10,
            "min_freq_exp": 0.0,
            "max_freq_exp": 8.0,
            "include_input": True,
        }
    )

    def __post_init__(self):
        if self.diffusion_config_path is not None:
            self.diffusion_config_path = Path(self.diffusion_config_path)
        if self.diffusion_ckpt_path is not None:
            self.diffusion_ckpt_path = Path(self.diffusion_ckpt_path)


class NerfusModel(Model):
    config: NerfusModelConfig

    def __init__(
        self,
        config: NerfusModelConfig,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(config=config, device=device, **kwargs)
        self._device = torch.device(device)

        self.position_encoding = SinusoidalEncoding(**config.position_encoding_config)

        self.field = NerfusField(
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
            cube_resolution=config.cube_resolution,
            input_dim=self.position_encoding.get_out_dim(),
        )

        if config.diffusion_config_path and config.diffusion_ckpt_path:
            self.diffusion_model = self.load_diffusion_model(
                config.diffusion_config_path, config.diffusion_ckpt_path
            )
        else:
            self.diffusion_model = None

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        param_groups = {"fields": list(self.parameters())}
        return param_groups

    def load_diffusion_model(self, diffusion_config_path, diffusion_ckpt_path):
        config = yaml.load(open(diffusion_config_path, "r"), Loader=yaml.Loader)
        config = DotMap(config)
        model = NerfusTrainer(config)
        ckpt = torch.load(diffusion_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        model = model.to(self.device)
        model.noise_scheduler.alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(
            self.device
        )
        model.dsds_loss.alphas = model.dsds_loss.alphas.to(self.device)

        print("Loaded diffusion config from", diffusion_config_path)
        print("Loaded diffusion checkpoint from", diffusion_ckpt_path)
        return model

    def get_denoised_border_prob(self, encoded_positions: torch.Tensor) -> torch.Tensor:
        if self.diffusion_model is None:
            return torch.zeros_like(encoded_positions[..., 0])

        with torch.no_grad():
            original_shape = encoded_positions.shape
            positions = encoded_positions.reshape(-1, original_shape[-1])

            elements_per_cube = self.config.cube_resolution**3
            num_cubes = positions.shape[0] // elements_per_cube

            positions = positions[: num_cubes * elements_per_cube]

            positions = positions.reshape(
                -1,
                1,
                self.config.cube_resolution,
                self.config.cube_resolution,
                self.config.cube_resolution,
            )

            noise = torch.randn_like(positions).to(self._device)
            t = (
                torch.ones(positions.shape[0], dtype=torch.long).to(self._device)
                * self.config.singlestep_starting_t
            )

            noisy_positions = positions + noise
            noise_pred = self.diffusion_model.model(noisy_positions, t)
            denoised = self.diffusion_model.noise_scheduler.step(
                model_output=noise_pred, timestep=t[0], sample=noisy_positions
            ).prev_sample
            border_prob = torch.sigmoid(denoised[..., 1])
            border_prob = border_prob.reshape(original_shape[:-1])

        return border_prob

    def get_denoised_scatter_density(
        self, encoded_positions: torch.Tensor
    ) -> torch.Tensor:
        if self.diffusion_model is None:
            return torch.zeros_like(encoded_positions[..., 0])

        with torch.no_grad():
            original_shape = encoded_positions.shape
            positions = encoded_positions.reshape(-1, original_shape[-1])

            elements_per_cube = self.config.cube_resolution**3
            num_cubes = positions.shape[0] // elements_per_cube

            positions = positions[: num_cubes * elements_per_cube]

            positions = positions.reshape(
                -1,
                1,
                self.config.cube_resolution,
                self.config.cube_resolution,
                self.config.cube_resolution,
            )

            noise = torch.randn_like(positions).to(self._device)
            t = (
                torch.ones(positions.shape[0], dtype=torch.long).to(self._device)
                * self.config.singlestep_starting_t
            )

            noisy_positions = positions + noise
            noise_pred = self.diffusion_model.model(noisy_positions, t)
            denoised = self.diffusion_model.noise_scheduler.step(
                model_output=noise_pred, timestep=t[0], sample=noisy_positions
            ).prev_sample
            scatter_density = torch.sigmoid(denoised[..., 2])
            scatter_density = scatter_density.reshape(original_shape[:-1])

        return scatter_density

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        origins = ray_bundle.origins
        directions = ray_bundle.directions
        starts = ray_bundle.nears
        ends = ray_bundle.fars

        mid = (starts + ends) / 2.0
        mid = mid.unsqueeze(-1)
        positions = origins + directions * mid

        encoded = self.position_encoding(positions)

        return {"encoded_positions": encoded}

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics_dict = {}
        if "ray_samples" in outputs:
            metrics_dict["num_samples"] = outputs[
                "ray_samples"
            ].frustums.lengths.numel()
        if "border_prob" in outputs:
            metrics_dict["border_prob_mean"] = outputs["border_prob"].mean()
        if "scatter_density" in outputs:
            metrics_dict["scatter_density_mean"] = outputs["scatter_density"].mean()
        return metrics_dict


@dataclass
class UltrasoundDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: UltrasoundDataParser)
    data: Path = Path("data/ultrasound/")
    scale_factor: float = 1.0
    """Scale the size of the scene."""
    aabb_scale: float = 1.0
    """Scale the size of the scene's bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""


class UltrasoundDataParser(DataParser):
    config: UltrasoundDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        transforms_path = self.config.data / "transforms.json"
        if not transforms_path.exists():
            raise ValueError(f"transforms.json not found at {transforms_path}")

        with transforms_path.open("r") as f:
            meta = json.load(f)

        image_filenames = []
        poses = []

        for frame in meta["frames"]:
            fname = Path(frame["file_path"])
            if not fname.suffix:
                fname = fname.with_suffix(".png")
            image_filenames.append(self.config.data / fname)

            poses.append(np.array(frame["transform_matrix"]))

        poses = np.array(poses).astype(np.float32)
        num_images = len(image_filenames)

        num_train_images = int(num_images * self.config.train_split_fraction)
        indices = np.arange(num_images)
        np.random.shuffle(indices)

        if split == "train":
            indices = indices[:num_train_images]
        else:
            indices = indices[num_train_images:]

        image_filenames = [image_filenames[i] for i in indices]
        poses = poses[indices]

        images = []
        for image_filename in image_filenames:
            if not image_filename.exists():
                raise ValueError(f"Image {image_filename} does not exist.")
            images.append(np.array(Image.open(image_filename)))
        images = np.stack(images)

        poses[:, :3, 3] *= self.config.scale_factor

        poses = torch.from_numpy(poses)

        camera_angle_x = float(meta.get("camera_angle_x", 0.8))  # Default FOV 45

        height, width = images[0].shape[:2]
        focal_length = float(0.5 * width / np.tan(0.5 * camera_angle_x))

        cx = width / 2.0
        cy = height / 2.0
        fx = focal_length
        fy = focal_length

        fx = torch.tensor(fx, dtype=torch.float32)
        fy = torch.tensor(fy, dtype=torch.float32)
        cx = torch.tensor(cx, dtype=torch.float32)
        cy = torch.tensor(cy, dtype=torch.float32)

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        aabb = (
            torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)
            * self.config.aabb_scale
        )
        scene_box = SceneBox(aabb=aabb)

        dataparser_outputs = DataparserOutputs(
            image_filenames=[str(x) for x in image_filenames],
            cameras=cameras,
            scene_box=scene_box,
            metadata={
                "height": height,
                "width": width,
                "focal_length": focal_length,
                "camera_angle_x": camera_angle_x,
                "num_images": len(poses),
                "split": split,
            },
        )

        CONSOLE.log(f"Loaded {len(image_filenames)} {split} images")
        return dataparser_outputs


nerfus_config = MethodSpecification(
    TrainerConfig(
        method_name="nerfus",
        project_name="nerfus-project",
        steps_per_eval_batch=1000,
        steps_per_eval_image=1000,
        steps_per_save=5000,
        steps_per_eval_all_images=0,
        save_only_latest_checkpoint=False,
        max_num_iterations=5001,
        mixed_precision=True,
        pipeline=NerfusPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=UltrasoundDataParserConfig(
                    data=None,
                    scale_factor=0.001,
                    aabb_scale=1.5,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfusModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                hidden_dim=256,
                hidden_layers=8,
                num_samples=64,
                use_spatial_distortion=True,
                diffusion_config_path=Path("config/synthetic-knee.yaml"),
                diffusion_ckpt_path=Path(
                    "outputs/diffusion/cubes_shapenet/ddpm-fulldata/checkpoints/synkn.ckpt"
                ),
                singlestep_starting_t=10,
                cube_resolution=32,
            ),
            probe_depth=0.14,
            probe_width=0.08,
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, websocket_port=None),
        vis="viewer",
    ),
    description="Uses the Nerfus pipeline.",
)
