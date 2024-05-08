import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Type

import imageio
import mediapy as media
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from dotmap import DotMap
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.fields.visibility_field import VisibilityField
from nerfstudio.models.vannila_nerf import NeRFModel, NeRFModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.render import render_rays
from nerfstudio.utils import (
    batchify,
    get_embedder,
    get_rays_us_linear,
    profiler,
    writer,
)
from torchtyping import TensorType
from typing_extensions import Literal

from nerfus.cubes.visualize3D import get_image_grid
from nerfus.lightning.nerfus_trainer import NerfusTrainer
from nerfus.nerf.nerfbusters_utils import random_train_pose
from nerfus.utils.visualizations import get_3Dimage_fast

print_stats = lambda x: print(
    f"x mean {x.mean():.3f}, std {x.std():.3f}, min {x.min():.3f}, max {x.max():.3f}"
)


def sample_cubes(
    min_x,
    min_y,
    min_z,
    max_x,
    max_y,
    max_z,
    res,
    spr_min,
    spr_max,
    num_cubes,
    cube_centers_x=None,
    cube_centers_y=None,
    cube_centers_z=None,
    device=None,
):
    assert device is not None, "device must be specified"

    # create the cubes
    scales = torch.rand(num_cubes, device=device) * (spr_max - spr_min) + spr_min
    cube_len = (max_x - min_x) * scales
    half_cube_len = cube_len / 2
    if cube_centers_x is None:
        cube_centers_x = (
            torch.rand(num_cubes, device=device) * (max_x - min_x - 2.0 * half_cube_len)
            + min_x
            + half_cube_len
        )
        cube_centers_y = (
            torch.rand(num_cubes, device=device) * (max_y - min_y - 2.0 * half_cube_len)
            + min_y
            + half_cube_len
        )
        cube_centers_z = (
            torch.rand(num_cubes, device=device) * (max_z - min_z - 2.0 * half_cube_len)
            + min_z
            + half_cube_len
        )
    else:
        assert cube_centers_x.shape == (num_cubes,)
        cube_centers_x = (
            cube_centers_x * (max_x - min_x - 2.0 * half_cube_len)
            + min_x
            + half_cube_len
        )
        cube_centers_y = (
            cube_centers_y * (max_y - min_y - 2.0 * half_cube_len)
            + min_y
            + half_cube_len
        )
        cube_centers_z = (
            cube_centers_z * (max_z - min_z - 2.0 * half_cube_len)
            + min_z
            + half_cube_len
        )
    cube_start_x = cube_centers_x - half_cube_len
    cube_start_y = cube_centers_y - half_cube_len
    cube_start_z = cube_centers_z - half_cube_len
    cube_start_xyz = torch.stack(
        [cube_start_x, cube_start_y, cube_start_z], dim=-1
    ).reshape(num_cubes, 1, 1, 1, 3)
    cube_end_x = cube_centers_x + half_cube_len
    cube_end_y = cube_centers_y + half_cube_len
    cube_end_z = cube_centers_z + half_cube_len
    cube_end_xyz = torch.stack([cube_end_x, cube_end_y, cube_end_z], dim=-1).reshape(
        num_cubes, 1, 1, 1, 3
    )
    l = torch.linspace(0, 1, res, device=device)
    xyz = torch.stack(
        torch.meshgrid([l, l, l], indexing="ij"), dim=-1
    )  # (res, res, res, 3)
    xyz = xyz[None, ...] * (cube_end_xyz - cube_start_xyz) + cube_start_xyz
    return xyz, cube_start_xyz, cube_end_xyz, scales


class WeightGrid(torch.nn.Module):
    """Keep track of the weights."""

    def __init__(self, resolution: int):
        super().__init__()
        self.resolution = resolution
        self.register_buffer(
            "_weights", torch.ones(self.resolution, self.resolution, self.resolution)
        )

    def update(
        self,
        xyz: TensorType["num_points", 3],
        weights: TensorType["num_points", 1],
        ema_decay: float = 0.5,
    ):
        """Updates the weights of the grid with EMA."""

        # xyz points should be in range [0, 1]
        assert xyz.min() >= 0, f"xyz min {xyz.min()}"
        assert xyz.max() < 1, f"xyz max {xyz.max()}"

        # verify the shapes are correct
        assert len(xyz.shape) == 2
        assert xyz.shape[0] == weights.shape[0]
        assert xyz.shape[1] == 3
        assert weights.shape[1] == 1

        # update the weights
        # indices = (xyz * self.resolution).long()
        # self._weights[indices[:, 0], indices[:, 1], indices[:, 2]] = torch.maximum(
        #     self._weights[indices[:, 0], indices[:, 1], indices[:, 2]] * ema_decay, weights.squeeze(-1)
        # )
        self._weights = ema_decay * self._weights
        indices = (xyz * self.resolution).long()
        self._weights[indices[:, 0], indices[:, 1], indices[:, 2]] += weights.squeeze(
            -1
        )

    def sample(
        self, num_points: int, randomize: bool = True
    ) -> TensorType["num_points", 3]:
        """Samples points from the grid where the value is above the threshold."""

        device = self._weights.device

        probs = self._weights.view(-1) / self._weights.view(-1).sum()
        dist = torch.distributions.categorical.Categorical(probs)
        sample = dist.sample((num_points,))

        h = torch.div(sample, self.resolution**2, rounding_mode="floor")
        d = sample % self.resolution
        w = torch.div(sample, self.resolution, rounding_mode="floor") % self.resolution

        idx = torch.stack([h, w, d], dim=1).float()
        if randomize:
            return (idx + torch.rand_like(idx).to(device)) / self.resolution
        else:
            return idx / self.resolution


@dataclass
class NerfusPipelineConfig(VanillaPipelineConfig):
    """Nerfus Pipeline Config"""

    _target: Type = field(default_factory=lambda: NerfusPipeline)

    lambda_b = 0.4
    lambda_s = 0.2

    # some default overrides

    # NeRF checkpoint
    nerf_checkpoint_path: Optional[Path] = None

    # 3D diffusion model
    diffusioncube_config_path: Optional[Path] = Path("config/synthetic-knee.yaml")
    diffusioncube_ckpt_path: Optional[Path] = Path(
        "outputs/diffusion/cubes_shapenet/ddpm-fulldata/checkpoints/synkn.ckpt"
    )

    # visualize options
    # what to visualize
    visualize_weights: bool = False
    visualize_cubes: bool = False
    visualize_patches: bool = False
    # how often to visualize
    steps_per_visualize_weights: int = 100
    steps_per_visualize_cubes: int = 1
    steps_per_visualize_patches: int = 10

    # cube sampling
    num_cubes: int = 40
    """Number of cubes per batch for training"""
    cube_resolution: int = 32
    cube_start_step: int = 0
    cube_scale_perc_range: Tuple[float, ...] = (
        0.01,
        0.10,
    )  # percentage of the scene box
    steps_per_draw_cubes: int = 20
    sample_method: Literal["uniform", "importance", "random", "fixed"] = "random"
    fix_samples_for_steps: int = 1
    num_views_per_cube: int = 3
    max_num_cubes_to_visualize: int = 6
    """If we should fix the batch of samples for a couple of steps."""
    fixed_cubes_center: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.0, -0.15),  # picnic - vase
        # (-0.85, 0.85, -0.5),  # plant - vase
        # (0.03, 0.28, -0.23), # table - chair
        # (-0.07, 0.02, -0.36),  # table - table
    )
    fixed_cubes_scale_perc: Tuple[Tuple[float], ...] = ((0.02),)
    aabb_scalar: float = 1.5

    # weight grid settings
    weight_grid_resolution: int = 100
    weight_grid_quantity: Literal["weights", "densities", "visibility"] = "weights"
    weight_grid_quantity_idx: int = -1
    weight_grid_update_per_step: int = 10

    # density to x
    density_to_x_crossing: float = 0.01
    density_to_x_max: float = 500.0
    density_to_x_activation: Literal[
        "sigmoid",
        "clamp",
        "sigmoid_complex",
        "rescale_clamp",
        "piecewise_linear",
        "batchnorm",
        "meannorm",
        "piecewise_loglinear",
        "piecewise_loglinear_sigmoid",
        "binarize",
        "piecewise_loglinear_threshold",
    ] = "binarize"
    piecewise_loglinear_threshold: float = 1e-3

    # TODO: add noise to densities (from original nerf paper)

    # patch sampling
    num_patches: int = 10
    """Number of patches per batch for training"""
    patch_resolution: int = 32
    """Patch resolution, where DiffRF used 48x48 and RegNeRF used 8x8"""
    focal_range: Tuple[float, float] = (3.0, 3.0)
    """Range of focal length"""
    central_rotation_range: Tuple[float, float] = (-180, 180)
    """Range of central rotation"""
    vertical_rotation_range: Tuple[float, float] = (-90, 20)
    """Range of vertical rotation"""
    jitter_std: float = 0.05
    """Std of camera direction jitter, so we don't just point the cameras towards the center every time"""
    center: Tuple[float, float, float] = (0, 0, 0)
    """Center coordinate of the camera sphere"""

    # -------------------------------------------------------------------------
    # 2D losses

    # regnerf loss
    use_regnerf_loss: bool = False
    regnerf_loss_mult: float = 1e-7

    # -------------------------------------------------------------------------
    # 3D losses

    # cube loss
    use_cube_loss: bool = False
    cube_loss_mult: float = 1e-1
    cube_loss_trange: Tuple[int, ...] = (9, 10)

    # multistep cube loss
    use_multistep_cube_loss: bool = False
    multistep_cube_loss_mult: float = 1e-2
    multistep_range: Tuple[int, ...] = (100, 700)
    num_multistep: int = 10

    # singlestep cube loss
    use_singlestep_cube_loss: bool = True
    singlestep_cube_loss_mult: float = 1e-1
    singlestep_target: float = 0.0
    singlestep_starting_t: int = 10

    # sparsity loss
    use_sparsity_loss: bool = False
    sparsity_loss_mult: float = 1e-2
    sparsity_length: float = 0.05

    # TV loss
    use_total_variation_loss: bool = False
    total_variation_loss_mult: float = 1e-7

    # visibility loss
    use_visibility_loss: bool = True
    visibility_loss_quantity: Literal["weights", "densities"] = "densities"
    visibility_loss_mult: float = 1e-6
    visibility_min_views: int = 1
    """Minimum number of training views that must be seen."""
    visibility_num_rays: int = 10

    # threshold loss
    use_threshold_loss: bool = False
    threshold_loss_mult: float = 1e-1


class Nerfus(VanillaPipeline):
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
        self.device = device
        self.aabb = (
            self.datamanager.train_dataset.scene_box.aabb.to(self.device)
            * self.config.aabb_scalar
        )
        self.weight_grid = WeightGrid(resolution=self.config.weight_grid_resolution).to(
            self.device
        )
        self.model = NeRFModel(NeRFModelConfig()).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lrate)

        # keep track of these to visualize cubes
        self.cube_start_xyz = None
        self.cube_end_xyz = None

        if self.config.nerf_checkpoint_path is not None:
            loaded_state = torch.load(
                self.config.nerf_checkpoint_path, map_location="cpu"
            )
            # remove any keys with diffusion
            for key in list(loaded_state["pipeline"].keys()):
                if "diffusion" in key:
                    del loaded_state["pipeline"][key]
                if "weight_grid" in key:
                    del loaded_state["pipeline"][key]
            self.load_state_dict(loaded_state["pipeline"], strict=False)
            print("Loaded NeRF checkpoint from", self.config.nerf_checkpoint_path)

        # load 3D diffusion model
        self.diffusioncube_model = self.load_diffusion_model(
            self.config.diffusioncube_config_path, self.config.diffusioncube_ckpt_path
        )

        # because these are not registered as parameters, we not to convert to device manually
        self.diffusioncube_model.noise_scheduler.betas = (
            self.diffusioncube_model.noise_scheduler.betas.to(self.device)
        )
        self.diffusioncube_model.noise_scheduler.alphas = (
            self.diffusioncube_model.noise_scheduler.alphas.to(self.device)
        )
        self.diffusioncube_model.noise_scheduler.alphas_cumprod = (
            self.diffusioncube_model.noise_scheduler.alphas_cumprod.to(self.device)
        )

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

    def render_rays(self, ray_batch, N_samples):
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
        bounds = ray_batch[..., 6:8].view(-1, 1, 2)
        near, far = bounds[..., 0], bounds[..., 1]
        t_vals = torch.linspace(0.0, 1.0, N_samples).to(self.device)
        z_vals = near * (1.0 - t_vals) + far * t_vals
        z_vals = z_vals.expand(N_rays, N_samples)
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

        raw = self.model(pts)
        ret = self.render_method_convolutional_ultrasound(raw, z_vals)
        return ret

    def render_method_convolutional_ultrasound(self, raw, z_vals):
        dists = torch.abs(z_vals[..., :-1] - z_vals[..., 1:])
        dists = torch.cat([dists, dists[..., -1:]], dim=-1)

        attenuation_coeff = torch.abs(raw[..., 0])
        attenuation = torch.exp(-attenuation_coeff * dists)
        attenuation_transmission = torch.cumprod(attenuation, dim=-1, exclusive=True)

        prob_border = torch.sigmoid(raw[..., 2])
        border_indicator = torch.bernoulli(prob_border)
        reflection_coeff = torch.sigmoid(raw[..., 1])
        reflection_transmission = 1.0 - reflection_coeff * border_indicator
        reflection_transmission = torch.cumprod(
            reflection_transmission, dim=-1, exclusive=True
        )

        border_convolution = torch.nn.functional.conv2d(
            border_indicator.unsqueeze(0).unsqueeze(0), self.g_kernel, padding="same"
        )
        border_convolution = border_convolution.squeeze()

        density_coeff_value = torch.sigmoid(raw[..., 3])
        scatter_density_distribution = torch.bernoulli(density_coeff_value)
        amplitude = torch.sigmoid(raw[..., 4])
        scatterers_map = scatter_density_distribution * amplitude

        psf_scatter = torch.nn.functional.conv2d(
            scatterers_map.unsqueeze(0).unsqueeze(0), self.g_kernel, padding="same"
        )
        psf_scatter = psf_scatter.squeeze()

        transmission = attenuation_transmission * reflection_transmission
        b = transmission * psf_scatter
        r = transmission * reflection_coeff * border_convolution

        intensity_map = b + r
        return {
            "intensity_map": intensity_map,
            "attenuation_coeff": attenuation_coeff,
            "reflection_coeff": reflection_coeff,
            "attenuation_transmission": attenuation_transmission,
            "reflection_transmission": reflection_transmission,
            "scatterers_density": scatter_density_distribution,
            "scatterers_density_coeff": density_coeff_value,
            "scatter_amplitude": amplitude,
            "b": b,
            "r": r,
            "transmission": transmission,
        }

    def query_field(self, xyz, method="border_probability"):
        raw = self.model(xyz)
        if method == "border_probability":
            return torch.sigmoid(raw[..., 2])
        elif method == "scatter_density":
            return torch.sigmoid(raw[..., 3])
        else:
            raise NotImplementedError(
                "Only border_probability and scatter_density are supported for now."
            )

    def apply_singlestep_cube_loss(
        self, step, x, border_probability, scatter_density, res, scales
    ):
        num_cubes = x.shape[0]
        x = x.reshape(num_cubes, 1, res, res, res)

        with torch.no_grad():
            xhat, w = self.diffusioncube_model.single_step_reverse_process(
                sample=x,
                starting_t=self.config.singlestep_starting_t,
                scale=scales,
            )

        xhat = torch.where(xhat < 0, -1, 1)
        mask_empty = xhat == -1
        mask_full = xhat == 1
        border_probability = border_probability.unsqueeze(1)
        scatter_density = scatter_density.unsqueeze(1)
        loss_border = (border_probability * mask_empty).sum()
        loss_border += (
            torch.clamp(self.config.singlestep_target - border_probability, 0)
            * mask_full
        ).sum()
        loss_scatter = (scatter_density * mask_empty).sum()
        loss_scatter += (
            torch.clamp(self.config.singlestep_target - scatter_density, 0) * mask_full
        ).sum()
        loss = self.config.lambda_b * loss_border + self.config.lambda_s * loss_scatter
        loss = loss / (border_probability.numel() + scatter_density.numel())
        return self.config.singlestep_cube_loss_mult * loss

    def apply_multistep_cube_loss(
        self, step: int, x, border_probability, scatter_density, res, scales
    ):
        num_cubes = x.shape[0]
        x = x.reshape(num_cubes, 1, res, res, res)

        min_step, max_step = self.config.cube_loss_trange
        num_multistep = self.config.num_multistep
        starting_t = (
            torch.randint(min_step, max_step, (1,)).to(self.device).long().item()
        )

        with torch.no_grad():
            xhat = self.diffusioncube_model.reverse_process(
                sample=x,
                scale=scales,
                bs=None,
                num_inference_steps=num_multistep,
                starting_t=starting_t,
            )

        xhat = torch.where(xhat < 0, -1, 1)
        mask_empty = xhat == -1
        mask_full = xhat == 1
        border_probability = border_probability.unsqueeze(1)
        scatter_density = scatter_density.unsqueeze(1)
        loss_border = (border_probability * mask_empty).sum()
        loss_border += (
            torch.clamp(self.config.singlestep_target - border_probability, 0)
            * mask_full
        ).sum()
        loss_scatter = (scatter_density * mask_empty).sum()
        loss_scatter += (
            torch.clamp(self.config.singlestep_target - scatter_density, 0) * mask_full
        ).sum()
        loss = self.config.lambda_b * loss_border + self.config.lambda_s * loss_scatter
        loss = loss / (border_probability.numel() + scatter_density.numel())
        return self.config.multistep_cube_loss_mult * loss

    def apply_threshold_loss(
        self, step, x, border_probability, scatter_density, res, scales
    ):
        target = 0.0
        xhat = torch.where(x < 0, -1, 1)
        mask_empty = xhat == -1
        mask_full = xhat == 1
        border_probability = border_probability.unsqueeze(1)
        scatter_density = scatter_density.unsqueeze(1)
        loss_border = (border_probability.abs() * mask_empty).sum()
        loss_border += (
            torch.clamp(target - border_probability.abs(), 0) * mask_full
        ).sum()
        loss_scatter = (scatter_density.abs() * mask_empty).sum()
        loss_scatter += (
            torch.clamp(target - scatter_density.abs(), 0) * mask_full
        ).sum()
        loss = self.config.lambda_b * loss_border + self.config.lambda_s * loss_scatter
        loss = loss / (border_probability.numel() + scatter_density.numel())
        return self.config.threshold_loss_mult * loss

    def apply_cube_loss(
        self, step, x, border_probability, scatter_density, res, scales
    ):
        num_cubes = x.shape[0]
        x = x.reshape(num_cubes, 1, res, res, res)

        min_t, max_t = self.config.cube_loss_trange
        timesteps = torch.randint(min_t, max_t, (1,)).to(self.device).long().item()

        model = self.diffusioncube_model.model
        dsds_loss = self.diffusioncube_model.dsds_loss
        grad = dsds_loss.grad_sds_unconditional(
            x, model, timesteps, scales, mult=self.config.cube_loss_mult
        )
        grad_mag = torch.mean(grad**2) ** 0.5

        loss_border = torch.sum(border_probability * grad_mag)
        loss_scatter = torch.sum(scatter_density * grad_mag)
        loss = self.config.lambda_b * loss_border + self.config.lambda_s * loss_scatter
        return loss / (border_probability.numel() + scatter_density.numel())

    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        # Populate the weight grid
        if step % self.config.weight_grid_update_per_step == 0:
            with torch.no_grad():
                ray_samples = model_outputs["ray_samples_list"][0]
                positions = ray_samples.frustums.get_positions()
                normalized_positions = SceneBox.get_normalized_positions(
                    positions, self.aabb
                )
                normalized_positions = normalized_positions.view(-1, 3)
                border_prob = model_outputs["border_probability"].view(-1, 1)
                scatter_density = model_outputs["scatter_density"].view(-1, 1)
                mask = (normalized_positions >= 0.0) & (normalized_positions < 1.0)
                mask = mask[:, 0] & mask[:, 1] & mask[:, 2]
                normalized_positions = normalized_positions[mask]
                border_prob = border_prob[mask]
                scatter_density = scatter_density[mask]
                self.weight_grid.update(
                    normalized_positions,
                    torch.clamp(border_prob + scatter_density, 0, 1),
                )

        # Incorporate the new losses
        if (
            self.config.use_singlestep_cube_loss
            or self.config.use_threshold_loss
            or self.config.use_cube_loss
            or self.config.use_multistep_cube_loss
        ):
            res = self.config.cube_resolution
            min_x, min_y, min_z = self.aabb[0]
            max_x, max_y, max_z = self.aabb[1]
            spr_min, spr_max = self.config.cube_scale_perc_range

            if step % self.config.fix_samples_for_steps == 0:
                self.xyz, self.cube_start_xyz, self.cube_end_xyz, self.scales = (
                    sample_cubes(
                        min_x,
                        min_y,
                        min_z,
                        max_x,
                        max_y,
                        max_z,
                        res,
                        spr_min,
                        spr_max,
                        self.config.num_cubes,
                        device=self.device,
                    )
                )

            self.xyz = self.xyz.to(self.device)
            border_probability = self.query_field(
                self.xyz, method="border_probability"
            ).squeeze(-1)
            scatter_density = self.query_field(
                self.xyz, method="scatter_density"
            ).squeeze(-1)

            if self.config.use_singlestep_cube_loss:
                loss = self.apply_singlestep_cube_loss(
                    step,
                    self.xyz,
                    border_probability,
                    scatter_density,
                    res,
                    self.scales,
                )
                loss_dict["ss_cube_loss"] = loss

            if self.config.use_threshold_loss:
                loss = self.apply_threshold_loss(
                    step,
                    self.xyz,
                    border_probability,
                    scatter_density,
                    res,
                    self.scales,
                )
                loss_dict["threshold_loss"] = loss

            if self.config.use_cube_loss:
                loss = self.apply_cube_loss(
                    step,
                    self.xyz,
                    border_probability,
                    scatter_density,
                    res,
                    self.scales,
                )
                loss_dict["cube_loss"] = loss

            if self.config.use_multistep_cube_loss:
                loss = self.apply_multistep_cube_loss(
                    step,
                    self.xyz,
                    border_probability,
                    scatter_density,
                    res,
                    self.scales,
                )
                loss_dict["ms_cube_loss"] = loss

        return model_outputs, loss_dict, metrics_dict

    def get_training_callbacks(self, training_callback_attributes):
        return super().get_training_callbacks(training_callback_attributes)
